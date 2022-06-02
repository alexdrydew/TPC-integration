"""Airflow graph for converting, uploading and registering the model.

Performs three main operations:
* TensorFlow -> ONNX model conversion (using tf2onnx). Executed in TPC-FastSim Docker container.
* Registering and uploading of the converted model to MLflow Model Registry.
* Triggers evaluation graph defined in evaluate_model.py on successful completion.

Parameters:
    saved_model_dir: a prefix in S3 with a model to export.
    model_name: desired model name in MLflow Model Registry. If the model is present in registry, the version will be
        incremented.
"""


import pendulum
from airflow import DAG
from airflow.models import Param
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from util import (
    DagRunParam,
    DagRunParamsDict,
    DockerOperatorExtended,
    DockerOperatorRemoteMapping,
    USER_DEFINED_MACROS,
    Constants,
)

DEFAULT_ARGS = {
    "start_date": pendulum.now(),
}


def upload_dataset_parameters_to_model(
    model_name, saved_model_dir, model_version, mlflow_host, saved_models_bucket
):
    """Task for saving dataset parameters used during training with model in MLflow Model Registry.

    Parameters are expected to be saved in intermediate trained model representation root as dataset.yaml.

    Args:
        model_name: model name in Mlflow Model Registry.
        saved_model_dir: source directory of trained model intermediate representation.
        model_version: model version in Mlflow Model Registry.
        mlflow_host: MLflow URL.
        saved_models_bucket: S3 bucket with trained model intermediate representations.
    """

    import mlflow
    from urllib.parse import urlparse
    from airflow.providers.amazon.aws.hooks.s3 import S3Hook

    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_host)
    uri = client.get_model_version_download_uri(model_name, model_version)
    parsed_uri = urlparse(uri)

    bucket = parsed_uri.netloc
    path = parsed_uri.path.lstrip("/")

    s3_hook = S3Hook(aws_conn_id="S3")
    s3_hook.copy_object(
        source_bucket_name=saved_models_bucket,
        dest_bucket_name=bucket,
        source_bucket_key=f"done/{saved_model_dir}/dataset.yaml",
        dest_bucket_key=f"{path}/dataset.yaml",
    )


def get_last_version(model_name, mlflow_host):
    """Task for retrieving last version of the model in MLflow Model Registry.

    Args:
        model_name: model name in Mlflow Model Registry.
        mlflow_host: MLflow URL.
    """

    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_host)
    new_version = client.get_latest_versions(model_name)[0].version

    return new_version


def has_dataset_parameters(saved_model_dir, saved_models_bucket):
    """Task to be used in BranchPythonOperator. Returns upload-dataset-parameters in case of dataset parameters being
    saved with the model, otherwise returns dataset-parameters-unknown.

    Args:
        saved_model_dir: source directory of trained model intermediate representation.
        saved_models_bucket: S3 bucket with trained model intermediate representations.

    Returns:

    """

    from airflow.providers.amazon.aws.hooks.s3 import S3Hook

    s3_hook = S3Hook(aws_conn_id="S3")
    if s3_hook.check_for_key(
        f"done/{saved_model_dir}/dataset.yaml", saved_models_bucket
    ):
        return "upload-dataset-parameters"
    else:
        return "dataset-parameters-unknown"


def create_convert_and_upload_model_dag(dag_id="convert-and-upload-model"):

    params = DagRunParamsDict(
        DagRunParam(name="saved_model_dir", dag_param=Param(default=str)),
        DagRunParam(name="model_name", dag_param=Param(default=str)),
    )

    dag = DAG(
        dag_id,
        schedule_interval=None,
        catchup=False,
        default_args=DEFAULT_ARGS,
        params=params.dag_params_view(),
        user_defined_macros=USER_DEFINED_MACROS,
        render_template_as_native_obj=True,
    )

    with dag:
        convert_operator = DockerOperatorExtended(
            task_id="convert-and-upload",
            dag=dag,
            image="alexdrydew/tpc-trainer:latest",
            docker_url="unix://var/run/docker.sock",
            network_mode="airflow-network",
            remote_mappings=[
                DockerOperatorRemoteMapping(
                    bucket=Constants.saved_models_bucket,
                    remote_path=f'/done/{params["saved_model_dir"]}',
                    mount_path="/TPC-FastSim/saved_models",
                    sync_on_start=True,
                )
            ],
            command=[
                "sh",
                "-c",
                "python export_model.py "
                f'--checkpoint_name {params["model_name"]}_{params["saved_model_dir"]} '
                "--export_format onnx "
                "--upload_to_mlflow "
                f"--aws_access_key_id {Constants.s3_access_key} "
                f"--aws_secret_access_key {Constants.s3_secret_access_key} "
                f"--mlflow_url {Constants.mlflow_host} "
                f"--s3_url {Constants.s3_host} "
                f'--mlflow_model_name {params["model_name"]} ',
            ],
        )

        get_version_operator = PythonOperator(
            task_id="get-last-version",
            dag=dag,
            python_callable=get_last_version,
            op_kwargs={
                "model_name": params["model_name"],
                "mlflow_host": Constants.mlflow_host,
            },
        )

        has_dataset_params_operator = BranchPythonOperator(
            task_id="has-dataset-params",
            dag=dag,
            python_callable=has_dataset_parameters,
            op_kwargs={
                "saved_model_dir": params["saved_model_dir"],
                "saved_models_bucket": Constants.saved_models_bucket,
            },
        )

        dataset_parameters_unknown_operator = EmptyOperator(
            task_id="dataset-parameters-unknown", dag=dag
        )

        upload_dataset_parameters_to_model_operator = PythonOperator(
            task_id="upload-dataset-parameters",
            dag=dag,
            python_callable=upload_dataset_parameters_to_model,
            op_kwargs={
                "model_name": params["model_name"],
                "saved_model_dir": params["saved_model_dir"],
                "model_version": '{{ task_instance.xcom_pull("get-last-version") }}',
                "mlflow_host": Constants.mlflow_host,
                "saved_models_bucket": Constants.saved_models_bucket,
            },
        )

        trigger_evaluation_operator = TriggerDagRunOperator(
            task_id="trigger-evaluate-model",
            dag=dag,
            trigger_dag_id="evaluate-model",
            trigger_run_id="{{ dag_run.run_id  }}",
            conf={
                "model_name": params["model_name"],
                "model_version": '{{ task_instance.xcom_pull("get-last-version") }}',
            },
            trigger_rule="one_success",
        )

        # save with dataset parameters branch
        (
            convert_operator
            >> get_version_operator
            >> has_dataset_params_operator
            >> upload_dataset_parameters_to_model_operator
            >> trigger_evaluation_operator
        )

        # no dataset parameters branch
        (
            get_version_operator
            >> has_dataset_params_operator
            >> dataset_parameters_unknown_operator
            >> trigger_evaluation_operator
        )

    return dag


dag = create_convert_and_upload_model_dag()
