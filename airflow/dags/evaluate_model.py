from pathlib import Path

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Param
from util import (
    Constants,
    USER_DEFINED_MACROS,
    DockerOperatorExtended,
    DockerOperatorRemoteMapping,
    DagRunParamsDict,
    DagRunParam,
)

SHARED_FOLDER = Path("/opt/airflow/shared")


DEFAULT_ARGS = {
    "start_date": pendulum.now(),
}


def update_mlflow_description(
    model_name,
    model_version,
    evaluation_results_bucket,
    evaluation_results_dir,
    mlflow_host,
    reverse_proxy_s3_host,
):
    import mlflow
    from io import StringIO

    def create_description():
        sio = StringIO()
        sio.write(
            f"![]({reverse_proxy_s3_host}/{evaluation_results_bucket}/{evaluation_results_dir}/test_results.png)\n"
        )
        sio.write("Some data...")
        return sio.getvalue()

    client = mlflow.tracking.MlflowClient(registry_uri=mlflow_host)
    client.update_model_version(
        name=model_name, version=model_version, description=create_description()
    )


def create_evaluate_model_dag(dag_id="evaluate-model"):

    params = DagRunParamsDict(
        DagRunParam(name="model_name", dag_param=Param(default="", type="string")),
        DagRunParam(name="model_version", dag_param=Param(default=1, type="integer")),
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

        evaluation_results_dir_template = (
            f'{params["model_name"]}_{params["model_version"]}'
        )

        evaluation_operator = DockerOperatorExtended(
            task_id="run-simulation",
            dag=dag,
            image="alexdrydew/mpdroot",
            docker_url="unix://var/run/docker.sock",
            network_mode="airflow-network",
            remote_mappings=[
                DockerOperatorRemoteMapping(
                    remote_path=f"/{evaluation_results_dir_template}",
                    mount_path="/evaluation_results",
                    bucket=Constants.evaluation_results_bucket,
                    sync_on_finish=True,
                )
            ],
            environment={
                "ONNX_MODEL_NAME": params["model_name"],
                "ONNX_MODEL_VERSION": params["model_version"],
                "MLFLOW_HOST": Constants.mlflow_hostname,
                "MLFLOW_PORT": Constants.mlflow_port,
                "S3_HOST": Constants.s3_hostname,
                "S3_PORT": Constants.s3_port,
            },
            command=[
                "sh",
                "-c",
                "cd macro/mpd && root -l -b -q runMC.C && root -l -b -q reco.C && "
                f"curl -o /evaluation_results/test_results.png http://mlflow.org/images/MLflow-logo-final-white-TM.png",
            ],
        )

        update_description = PythonOperator(
            task_id="update-mlflow-description",
            dag=dag,
            python_callable=update_mlflow_description,
            op_kwargs={
                "model_name": params["model_name"],
                "model_version": params["model_version"],
                "evaluation_results_bucket": Constants.evaluation_results_bucket,
                "evaluation_results_dir": evaluation_results_dir_template,
                "mlflow_host": Constants.mlflow_host,
                "reverse_proxy_s3_host": Constants.reverse_proxy_s3_host,
            },
        )

        evaluation_operator >> update_description

    return dag


dag = create_evaluate_model_dag()
