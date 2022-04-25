from pathlib import Path
import pendulum
from airflow.operators.dummy import DummyOperator

from util import DockerOperatorExtended, dict_hash, DockerOperatorRemoteMapping, get_aws_credentials

from airflow import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


SHARED_FOLDER = Path('/opt/airflow/shared')

DEFAULT_ARGS = {
    'start_date': pendulum.now(),
    'dataset_parameters': '{{ dag_run.conf["dataset_parameters"] }}',
    'dataset_dir': '{{ dict_hash(dag_run.conf["dataset_parameters"]) }}',
    'datasets_bucket': '{{ var.value.datasets_bucket }}',
}


def dataset_exists(**context):
    dataset_dir = Path(context['templates_dict']['dataset_dir'])
    bucket = context['templates_dict']['datasets_bucket']

    s3 = S3Hook(aws_conn_id='S3')

    if not s3.check_for_prefix(str(dataset_dir), '/', bucket_name=bucket):
        return 'create-dataset'
    return 'dataset-ready'


def create_get_dataset_operators(dag, args):
    dataset_exists_operator = BranchPythonOperator(
        task_id='check-dataset-exists',
        dag=dag,
        python_callable=dataset_exists,
        templates_dict=args,
        provide_context=True,
    )

    create_dataset_operator = DockerOperatorExtended(
        task_id='create-dataset',
        dag=dag,
        image='alexdrydew/mpdroot',
        docker_url='unix://var/run/docker.sock',
        output_remote_mappings=[DockerOperatorRemoteMapping(
            bucket='datasets', remote_path='/', mount_path='/remote_output'
        )],
        command=[
            # TODO: fake dataset generation
            'sh', '-c',
            f'git clone https://github.com/SiLiKhon/TPC-FastSim.git && '
            f'mkdir -p /remote_output/{args["dataset_dir"]}/raw && '
            f'cp -r TPC-FastSim/data/data_v4/raw/* /remote_output/{args["dataset_dir"]}/raw'
        ]
    )

    convert_dataset_operator = DockerOperatorExtended(
        task_id='convert-dataset',
        dag=dag,
        image='alexdrydew/tpc-trainer:latest',
        docker_url='unix://var/run/docker.sock',
        input_remote_mappings=[DockerOperatorRemoteMapping(
            bucket='datasets', remote_path=f'/{args["dataset_dir"]}', mount_path='/TPC-FastSim/data/data_v4'
        )],
        output_remote_mappings=[DockerOperatorRemoteMapping(
            bucket='datasets', remote_path=f'/{args["dataset_dir"]}', mount_path='/new_data'
        )],
        command=[
            '-c', '"from data import preprocessing; from pathlib import Path;'
                  "preprocessing._VERSION = 'data_v4';"
                  "output_path = Path('/new_data/csv/digits.csv');"
                  'output_path.parent.mkdir(exist_ok=True, parents=True);'
                  'preprocessing.raw_to_csv(fname_out=str(output_path));'
                  '"'
        ],
    )

    dataset_ready = DummyOperator(task_id='dataset-ready', dag=dag, trigger_rule='one_success')

    dataset_exists_operator >> create_dataset_operator >> convert_dataset_operator >> dataset_ready
    dataset_exists_operator >> dataset_ready

    return dataset_ready


def create_get_dataset_dag(args, dag_id='get-dataset', schedule_interval=None):
    args = {**DEFAULT_ARGS, **args}

    dag = DAG(
        dag_id,
        schedule_interval=schedule_interval,
        catchup=False,
        default_args=args,
        user_defined_macros={
            'dict_hash': dict_hash,
            'get_aws_credentials': get_aws_credentials
        },
    )

    with dag:
        create_get_dataset_operators(dag, args)

    return dag


dag = create_get_dataset_dag(DEFAULT_ARGS)
