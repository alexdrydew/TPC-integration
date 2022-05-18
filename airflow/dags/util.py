import dataclasses
import json
import os
import shutil
import tempfile
from pathlib import Path, PosixPath
from typing import Optional, Union, List

from airflow import AirflowException
from airflow.hooks.base import BaseHook
from airflow.providers.docker.operators.docker import DockerOperator, stringify
from airflow.utils.context import Context
from airflow.models import Param
from docker.types import DeviceRequest, Mount


def get_aws_credentials():
    from urllib.parse import urlparse

    credentials = json.loads(BaseHook.get_connection("S3").get_extra())
    parsed = urlparse(credentials["host"])
    credentials["hostname"] = parsed.hostname
    credentials["port"] = parsed.port
    return credentials


class Constants:
    airflow_bucket = "{{ var.value.airflow_bucket }}"
    saved_models_bucket = "{{ var.value.saved_models_bucket }}"
    datasets_bucket = "{{ var.value.datasets_bucket }}"
    evaluation_results_bucket = "{{ var.value.evaluation_results_bucket }}"
    tensorboard_bucket = "{{ var.value.tensorboard_bucket }}"
    s3_hostname = get_aws_credentials()["hostname"]
    s3_port = get_aws_credentials()["port"]
    s3_host = get_aws_credentials()["host"]
    mlflow_hostname = "{{ var.value.mlflow_host }}"
    mlflow_port = "{{ var.value.mlflow_tracking_port }}"
    mlflow_host = (
        "http://{{ var.value.mlflow_host }}:{{ var.value.mlflow_tracking_port }}"
    )
    s3_access_key = get_aws_credentials()["aws_access_key_id"]
    s3_secret_access_key = get_aws_credentials()["aws_secret_access_key"]
    reverse_proxy_host = "http://{{ var.value.reverse_proxy_host }}:{{ var.value.reverse_proxy_s3_port }}"


def dict_hash(params):
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_OID, json.dumps(params, sort_keys=True)))


USER_DEFINED_MACROS = {
    "dict_hash": dict_hash,
    "get_aws_credentials": get_aws_credentials,
}


class Templates:
    @classmethod
    def dataset_dir(cls, dataset_parameters_template: str):
        dataset_parameters_template = dataset_parameters_template.lstrip("{").rstrip(
            "}"
        )
        return f"{{{{ dict_hash({dataset_parameters_template}) }}}}"

    @classmethod
    def model_dir(cls, dataset_parameters_template: str, model_config_template: str):
        dataset_parameters_template = dataset_parameters_template.lstrip("{").rstrip(
            "}"
        )
        model_config_template = model_config_template.lstrip("{").rstrip("}")
        dict_template = f"{{'dataset_params': {dataset_parameters_template}, 'model_config': {model_config_template}}}"
        return f"{{{{ dict_hash({dict_template}) }}}}"


@dataclasses.dataclass
class DagRunParam:
    name: str
    dag_param: Param

    @property
    def template(self):
        return f'{{{{ dag_run.conf["{self.name}"] }}}}'


class DagRunParamsDict:
    def __init__(self, *params: DagRunParam):
        self.params = {param.name: param for param in params}

    def dag_params_view(self):
        return {
            param_name: param.dag_param for param_name, param in self.params.items()
        }

    def __getitem__(self, item):
        return self.params[item].template


def s3_to_local_folder(
    s3_conn_id: str, s3_bucket: str, s3_path: Union[PosixPath, str], local_path: Path
):
    from airflow.providers.amazon.aws.hooks.s3 import S3Hook

    s3 = S3Hook(s3_conn_id)
    s3_path = PosixPath(s3_path).relative_to("/")

    local_path.mkdir(parents=True, exist_ok=True)

    if s3.check_for_key(str(s3_path), bucket_name=s3_bucket):
        downloaded = s3.download_file(
            bucket_name=s3_bucket, key=str(s3_path), local_path=str(local_path)
        )
        (local_path / downloaded).rename(local_path / s3_path.name)
    elif s3.check_for_prefix(str(s3_path), "/", bucket_name=s3_bucket):
        keys = s3.list_keys(bucket_name=s3_bucket, prefix=str(s3_path))
        for key in keys:
            filepath = PosixPath(key).relative_to(s3_path)
            downloaded = s3.download_file(
                bucket_name=s3_bucket, key=key, local_path=str(local_path)
            )
            (local_path / filepath).parent.mkdir(exist_ok=True, parents=True)
            (local_path / downloaded).rename(local_path / filepath)
    else:
        raise ValueError(f"{s3_path} is neither a key nor a prefix in {s3_bucket}.")


def local_folder_to_s3(
    s3_conn_id: str,
    s3_bucket: str,
    s3_path: Union[PosixPath, str],
    local_path: Path,
    replace=True,
):
    from airflow.providers.amazon.aws.hooks.s3 import S3Hook

    s3 = S3Hook(s3_conn_id)
    s3_path = PosixPath(s3_path).relative_to("/")

    if local_path.is_dir():
        for sub_path in local_path.rglob("*.*"):
            key = s3_path / local_path.name / sub_path.relative_to(local_path)
            s3.load_file(
                filename=sub_path, key=str(key), bucket_name=s3_bucket, replace=replace
            )
    else:
        s3.load_file(
            filename=local_path,
            key=str(s3_path / local_path.name),
            bucket_name=s3_bucket,
            replace=replace,
        )


@dataclasses.dataclass
class DockerOperatorRemoteMapping:
    template_fields = ("bucket", "remote_path", "mount_path")

    bucket: str
    remote_path: str
    mount_path: str
    sync_on_start: bool = False
    sync_on_finish: bool = False


class DockerOperatorExtended(DockerOperator):
    template_fields = (*DockerOperator.template_fields, "remote_mappings", "mounts")

    def __init__(
        self,
        remote_mappings: List[DockerOperatorRemoteMapping] = None,
        device_requests: List[DeviceRequest] = None,
        map_output_on_fail=False,
        **kwargs,
    ):
        self.map_output_on_fail = map_output_on_fail
        self.remote_mappings = remote_mappings or []
        mounts = kwargs.get("mounts", [])

        mounts.append(
            Mount(
                source="/etc/passwd", target="/etc/passwd", read_only=True, type="bind"
            )
        )

        for mount in mounts:
            mount.template_fields = ("Source", "Target", "Type")
        kwargs["mounts"] = mounts

        self.device_requests = device_requests
        user = kwargs.pop("user", None)
        assert user is None, f"User is not supported in {type(self)}."
        kwargs["user"] = os.getuid()

        super().__init__(**kwargs)

    def execute(self, context: "Context") -> Optional[str]:
        initial_mounts = list(self.mounts)

        tmp_dirs_to_mappings = {}
        for mapping in self.remote_mappings:
            tmp_dir = Path(tempfile.mkdtemp(dir=self.host_tmp_dir))
            tmp_dirs_to_mappings[tmp_dir] = mapping
            if mapping.sync_on_start:
                s3_to_local_folder("S3", mapping.bucket, mapping.remote_path, tmp_dir)
            self.mounts.append(
                Mount(source=str(tmp_dir), target=str(mapping.mount_path), type="bind")
            )

        failed = True

        try:
            result = super().execute(context)
            failed = False
        finally:
            if not failed or self.map_output_on_fail:
                for tmp_dir, mapping in tmp_dirs_to_mappings.items():
                    if mapping.sync_on_finish:
                        for subfolder in tmp_dir.iterdir():
                            local_folder_to_s3(
                                "S3", mapping.bucket, mapping.remote_path, subfolder
                            )
                    shutil.rmtree(tmp_dir)

        self.mounts = initial_mounts
        return result

    def _run_image_with_mounts(
        self, target_mounts, add_tmp_variable: bool
    ) -> Optional[Union[List[str], str]]:
        if add_tmp_variable:
            self.environment["AIRFLOW_TMP_DIR"] = self.tmp_dir
        else:
            self.environment.pop("AIRFLOW_TMP_DIR", None)
        if not self.cli:
            raise Exception("The 'cli' should be initialized before!")
        self.container = self.cli.create_container(
            command=self.format_command(self.command),
            name=self.container_name,
            environment={**self.environment, **self._private_environment},
            host_config=self.cli.create_host_config(
                auto_remove=False,
                mounts=target_mounts,
                network_mode=self.network_mode,
                shm_size=self.shm_size,
                dns=self.dns,
                dns_search=self.dns_search,
                cpu_shares=int(round(self.cpus * 1024)),
                mem_limit=self.mem_limit,
                cap_add=self.cap_add,
                extra_hosts=self.extra_hosts,
                privileged=self.privileged,
                device_requests=self.device_requests,  # Override: devices support
            ),
            image=self.image,
            user=self.user,
            entrypoint=self.format_command(self.entrypoint),
            working_dir=self.working_dir,
            tty=self.tty,
        )
        logstream = self.cli.attach(
            container=self.container["Id"], stdout=True, stderr=True, stream=True
        )
        try:
            self.cli.start(self.container["Id"])

            log_lines = []
            for log_chunk in logstream:
                log_chunk = stringify(log_chunk).strip()
                log_lines.append(log_chunk)
                self.log.info("%s", log_chunk)

            result = self.cli.wait(self.container["Id"])
            if result["StatusCode"] != 0:
                joined_log_lines = "\n".join(log_lines)
                raise AirflowException(
                    f"Docker container failed: {repr(result)} lines {joined_log_lines}"
                )

            if self.retrieve_output:
                return self._attempt_to_retrieve_result()
            elif self.do_xcom_push:
                log_parameters = {
                    "container": self.container["Id"],
                    "stdout": True,
                    "stderr": True,
                    "stream": True,
                }
                try:
                    if self.xcom_all:
                        return [
                            stringify(line).strip()
                            for line in self.cli.logs(**log_parameters)
                        ]
                    else:
                        lines = [
                            stringify(line).strip()
                            for line in self.cli.logs(**log_parameters, tail=1)
                        ]
                        return lines[-1] if lines else None
                except StopIteration:
                    # handle the case when there is not a single line to iterate on
                    return None
            return None
        finally:
            if self.auto_remove:
                self.cli.remove_container(self.container["Id"])
