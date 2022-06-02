"""Airflow graph for dataset retrieval.

Checks if dataset exists for supplied parameters. If dataset is already saved simply returns it. Otherwise, starts
dataset generation process using MPDRoot Docker Container.

Parameters:
    dataset_parameters: a dictionary with arbitrary dataset parameters.
"""

import pendulum
from airflow.models import Param
from airflow.operators.empty import EmptyOperator

from util import (
    DockerOperatorExtended,
    DockerOperatorRemoteMapping,
    Constants,
    Templates,
    USER_DEFINED_MACROS,
    DagRunParam,
    DagRunParamsDict,
)

from airflow import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


DEFAULT_ARGS = {
    "start_date": pendulum.now(),
}


def dataset_exists(dataset_dir, datasets_bucket):
    """Checks if dataset exists. If dataset exists return "do-nothing", otherwise returns "create-dataset".

    Args:
        dataset_dir: S3 dataset objects prefix.
        datasets_bucket: S3 datasets bucket name.
    """

    s3 = S3Hook(aws_conn_id="S3")

    if not s3.check_for_prefix(str(dataset_dir), "/", bucket_name=datasets_bucket):
        return "create-dataset"
    return "do-nothing"


def create_get_dataset_operators(dag: DAG, params: DagRunParamsDict):
    """A helper function that inserts dataset creation tasks in the beginning of the supplied DAG. Returns last node in
    the graph."""

    dataset_dir_template = Templates.dataset_dir(params["dataset_parameters"])

    dataset_exists_operator = BranchPythonOperator(
        task_id="check-dataset-exists",
        dag=dag,
        python_callable=dataset_exists,
        op_kwargs={
            "dataset_dir": dataset_dir_template,
            "datasets_bucket": Constants.datasets_bucket,
        },
    )

    create_dataset_operator = DockerOperatorExtended(
        task_id="create-dataset",
        dag=dag,
        image="alexdrydew/mpdroot",
        docker_url="unix://var/run/docker.sock",
        remote_mappings=[
            DockerOperatorRemoteMapping(
                bucket=Constants.datasets_bucket,
                remote_path="/",
                mount_path="/remote_output",
                sync_on_finish=True,
            )
        ],
        command=[
            # TODO: fake dataset generation
            "sh",
            "-c",
            "git clone https://github.com/SiLiKhon/TPC-FastSim.git && "
            f"mkdir -p /remote_output/{dataset_dir_template}/raw && "
            f"cp -r TPC-FastSim/data/data_v4/raw/* /remote_output/{dataset_dir_template}/raw && "
            "chmod -R 777 /remote_output",
        ],
    )

    convert_dataset_operator = DockerOperatorExtended(
        task_id="convert-dataset",
        dag=dag,
        image="alexdrydew/tpc-trainer:latest",
        docker_url="unix://var/run/docker.sock",
        remote_mappings=[
            DockerOperatorRemoteMapping(
                bucket=Constants.datasets_bucket,
                remote_path=f"/{dataset_dir_template}",
                mount_path="/TPC-FastSim/data/data_v4",
                sync_on_start=True,
            ),
            DockerOperatorRemoteMapping(
                bucket=Constants.datasets_bucket,
                remote_path=f"/{dataset_dir_template}",
                mount_path="/new_data",
                sync_on_finish=True,
            ),
        ],
        command=[
            "sh",
            "-c",
            "python -c "
            '"'
            "from data import preprocessing; from pathlib import Path;"
            "preprocessing._VERSION = 'data_v4';"
            "output_path = Path('/new_data/csv/digits.csv');"
            "output_path.parent.mkdir(exist_ok=True, parents=True);"
            "preprocessing.raw_to_csv(fname_out=str(output_path));"
            '" && '
            "chmod -R 777 /new_data",
        ],
    )

    dataset_ready = EmptyOperator(
        task_id="dataset-ready", dag=dag, trigger_rule="one_success"
    )

    (
        dataset_exists_operator
        >> create_dataset_operator
        >> convert_dataset_operator
        >> dataset_ready
    )
    # "do-nothing" is needed to prevent skipping all downstream tasks in case of "create-dataset" task execution
    (
        dataset_exists_operator
        >> EmptyOperator(task_id="do-nothing", dag=dag)
        >> dataset_ready
    )

    return dataset_ready


def create_get_dataset_dag(dag_id="get-dataset", schedule_interval=None):

    params = DagRunParamsDict(
        DagRunParam(
            name="dataset_parameters", dag_param=Param(default=dict(), type="object")
        )
    )

    dag = DAG(
        dag_id,
        schedule_interval=schedule_interval,
        catchup=False,
        default_args=DEFAULT_ARGS,
        user_defined_macros=USER_DEFINED_MACROS,
        params=params.dag_params_view(),
        render_template_as_native_obj=True,
    )

    with dag:
        create_get_dataset_operators(dag, params)

    return dag


dag = create_get_dataset_dag()
