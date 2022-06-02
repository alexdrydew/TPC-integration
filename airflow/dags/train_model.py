"""Airflow graph for model training and dataset creation.

A combined graph of dataset creation and model training. The beginning of the graph is exactly the graph defined in
get_dataset.py. Trains neural TPC model with supplied parameters using generated dataset inside TPC-FastSim Docker
container.
Key features:
* Supports training resuming after fail.
* Supports training on both GPU and CPU.
* Saves training logs to S3 in real time.
If the training is successful triggers convert-and-upload-model graph defined in convert_and_upload_model.py

Parameters:
    ignore_saves: if True starts training from the beginning even if intermediate save for the model and dataset
        parameters is present.
    dataset_parameters: a dictionary with arbitrary dataset parameters.
    model_config: a dictionary with neural TPC model hyperparameters.
    model_name: desired model name in MLflow Model Registry. If the model is present in registry, the version will be
        incremented.
"""

import pendulum
import os
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import Param
from docker.types import DeviceRequest
from pathlib import Path
from get_dataset import create_get_dataset_operators
from util import (
    DockerOperatorExtended,
    DockerOperatorRemoteMapping,
    Constants,
    Templates,
    USER_DEFINED_MACROS,
    DagRunParam,
    DagRunParamsDict,
)


DEFAULT_ARGS = {
    "start_date": pendulum.now(),
}


def create_config(model_config, model_dir, airflow_bucket):
    """Converts model_config DAG parameter to yaml and saves it with the model.

    Args:
        model_config: dictionary with model hyperparameters.
        model_dir: trained model intermediate representations directory.
        airflow_bucket: S3 bucket name for temp files.
    """

    import yaml
    import tempfile
    from util import local_folder_to_s3

    with tempfile.TemporaryDirectory() as file_dir:
        file_dir = Path(file_dir)
        (file_dir / "models" / "configs").mkdir(exist_ok=True, parents=True)
        with open(file_dir / "models" / "configs" / "model.yaml", "w") as f:
            yaml.dump(model_config, f, default_flow_style=False)

        local_folder_to_s3(
            "S3",
            airflow_bucket,
            s3_path=f"/tmp/{model_dir}",
            local_path=file_dir / "models",
        )


def copy_model_to_done_folder(model_dir, saved_models_bucket, dataset_parameters):
    """Copies model with "intermediate" S3 prefix using "done" prefix.

    Args:
        model_dir: saved model S3 prefix.
        saved_models_bucket: S3 bucket name.
        dataset_parameters: parameters of the dataset used during training.
    """

    import yaml
    import tempfile
    from util import local_folder_to_s3

    s3_hook = S3Hook(aws_conn_id="S3")
    for key in s3_hook.list_keys(
        bucket_name=saved_models_bucket, prefix=f"intermediate/{model_dir}"
    ):
        s3_hook.copy_object(
            source_bucket_name=saved_models_bucket,
            dest_bucket_name=saved_models_bucket,
            source_bucket_key=key,
            dest_bucket_key=key.replace("intermediate", "done"),
        )

    with tempfile.TemporaryDirectory() as file_dir:
        file_dir = Path(file_dir)
        with open(file_dir / "dataset.yaml", "w") as f:
            yaml.dump(dataset_parameters, f, default_flow_style=False)

        local_folder_to_s3(
            "S3",
            saved_models_bucket,
            s3_path=f"/done/{model_dir}",
            local_path=file_dir / "dataset.yaml",
        )


def decide_if_continue(ignore_saves, model_dir, saved_models_bucket):
    """Checks if training should be resumed. Returns "prepare-config" if training should be started from the beginnning,
     otherwise returns "train-model-continue"

    Args:
        ignore_saves: if True always starts training from the beginning.
        model_dir: model saves directory.
        saved_models_bucket: S3 bucket name for intermediate model representations.
    """

    s3_hook = S3Hook(aws_conn_id="S3")

    if ignore_saves or not s3_hook.check_for_prefix(
        bucket_name=saved_models_bucket,
        prefix=f"intermediate/{model_dir}",
        delimiter="/",
    ):
        return "prepare-config"
    return "train-model-continue"


def create_train_model_dag(dag_id="train-model"):

    params = DagRunParamsDict(
        DagRunParam(
            name="ignore_saves", dag_param=Param(default=False, type="boolean")
        ),
        DagRunParam(
            name="dataset_parameters", dag_param=Param(default=dict(), type="object")
        ),
        DagRunParam(
            name="model_config", dag_param=Param(default=dict(), type="object")
        ),
        DagRunParam(name="model_name", dag_param=Param(default="", type="string")),
    )

    dag = DAG(
        dag_id,
        schedule_interval=None,
        catchup=False,
        default_args=DEFAULT_ARGS,
        user_defined_macros=USER_DEFINED_MACROS,
        params=params.dag_params_view(),
        render_template_as_native_obj=True,
    )

    with dag:

        dataset_dir_template = Templates.dataset_dir(params["dataset_parameters"])
        model_dir_template = Templates.model_dir(
            params["dataset_parameters"], params["model_config"]
        )

        get_dataset_operator = create_get_dataset_operators(dag, params)

        prepare_config_operator = PythonOperator(
            task_id="prepare-config",
            dag=dag,
            python_callable=create_config,
            op_kwargs={
                "model_config": params["model_config"],
                "model_dir": model_dir_template,
                "airflow_bucket": Constants.airflow_bucket,
            },
        )

        decide_if_continue_operator = BranchPythonOperator(
            task_id="decide-if-continue",
            dag=dag,
            python_callable=decide_if_continue,
            op_kwargs={
                "ignore_saves": params["ignore_saves"],
                "model_dir": model_dir_template,
                "saved_models_bucket": Constants.saved_models_bucket,
            },
        )

        common_train_args = {
            "dag": dag,
            "image": "alexdrydew/tpc-trainer",
            "docker_url": "unix://var/run/docker.sock",
            # GPU passthrough configuration
            "device_requests": [DeviceRequest(capabilities=[["gpu"]], count=1)]
            if os.getenv("TRAIN_ON_GPU") == "true"
            else [],
            # save to continue training later in case of fail
            "map_output_on_fail": True,
            "network_mode": "airflow-network",
            # S3 credentials for training logs
            "environment": {
                "AWS_ACCESS_KEY_ID": Constants.s3_access_key,
                "AWS_SECRET_ACCESS_KEY": Constants.s3_secret_access_key,
                "S3_ENDPOINT": f"{Constants.s3_hostname}:{Constants.s3_port}",
                "S3_USE_HTTPS": 0,
                "S3_VERIFY_SSL": 0,
            },
        }

        train_model_operator_from_start = DockerOperatorExtended(
            task_id="train-model-from-start",
            **common_train_args,
            remote_mappings=[
                DockerOperatorRemoteMapping(
                    bucket=Constants.airflow_bucket,
                    remote_path=f"/tmp/{model_dir_template}/models/configs",
                    mount_path="/TPC-FastSim/models/configs",
                    sync_on_start=True,
                ),
                DockerOperatorRemoteMapping(
                    bucket=Constants.datasets_bucket,
                    remote_path=f"/{dataset_dir_template}",
                    mount_path="/TPC-FastSim/data/data_v4",
                    sync_on_start=True,
                ),
                DockerOperatorRemoteMapping(
                    bucket=Constants.saved_models_bucket,
                    remote_path=f"/intermediate/{model_dir_template}",
                    mount_path="/TPC-FastSim/saved_models",
                    sync_on_finish=True,
                ),
            ],
            command=[
                "sh",
                "-c",
                f'python run_model_v4.py --checkpoint_name {params["model_name"]}_{model_dir_template} '
                f"--config models/configs/model.yaml --logging_dir s3://{Constants.tensorboard_bucket}/logs "
                f"--gpu_num " + ("0" if os.getenv("TRAIN_ON_GPU") == "true" else "''"),
            ],
        )
        train_model_operator_continue = DockerOperatorExtended(
            task_id="train-model-continue",
            **common_train_args,
            remote_mappings=[
                DockerOperatorRemoteMapping(
                    bucket=Constants.airflow_bucket,
                    remote_path=f"/tmp/{model_dir_template}/models/configs",
                    mount_path="/TPC-FastSim/models/configs",
                    sync_on_start=True,
                ),
                DockerOperatorRemoteMapping(
                    bucket=Constants.datasets_bucket,
                    remote_path=f"/{dataset_dir_template}",
                    mount_path="/TPC-FastSim/data/data_v4",
                    sync_on_start=True,
                ),
                DockerOperatorRemoteMapping(
                    bucket=Constants.saved_models_bucket,
                    remote_path=f"/intermediate/{model_dir_template}",
                    mount_path="/TPC-FastSim/saved_models",
                    sync_on_start=True,
                    sync_on_finish=True,
                ),
            ],
            command=[
                "sh",
                "-c",
                f'python run_model_v4.py --checkpoint_name {params["model_name"]}_{model_dir_template} '
                f"--logging_dir s3://{Constants.tensorboard_bucket}/logs "
                f"--gpu_num " + ("0" if os.getenv("TRAIN_ON_GPU") == "true" else "''"),
            ],
        )

        copy_trained_model_operator = PythonOperator(
            task_id="copy-to-done",
            dag=dag,
            python_callable=copy_model_to_done_folder,
            op_kwargs={
                "model_dir": model_dir_template,
                "saved_models_bucket": Constants.saved_models_bucket,
                "dataset_parameters": params["dataset_parameters"],
            },
            trigger_rule="one_success",
        )

        trigger_convert_model_dag = TriggerDagRunOperator(
            task_id="trigger-convert-model",
            dag=dag,
            trigger_dag_id="convert-and-upload-model",
            trigger_run_id="{{ dag_run.run_id  }}",
            conf={
                "saved_model_dir": model_dir_template,
                "model_name": params["model_name"],
            },
        )

        get_dataset_operator >> decide_if_continue_operator

        (
            decide_if_continue_operator
            >> prepare_config_operator
            >> train_model_operator_from_start
            >> copy_trained_model_operator
        )
        (
            decide_if_continue_operator
            >> train_model_operator_continue
            >> copy_trained_model_operator
        )
        copy_trained_model_operator >> trigger_convert_model_dag

    return dag


dag = create_train_model_dag()
