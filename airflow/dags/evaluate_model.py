from pathlib import Path, PosixPath

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from docker.types import Mount

from util import (local_folder_to_s3, dict_hash, DockerOperatorExtended,
                  DockerOperatorRemoteMapping, get_aws_credentials, COMMON_DEFAULT_ARGS)

SHARED_FOLDER = Path('/opt/airflow/shared')


DEFAULT_ARGS = {
    **COMMON_DEFAULT_ARGS,
    'start_date': pendulum.now(),
    'model_name': '{{ dag_run.conf["model_name"] }}',
    'model_version': '{{ dag_run.conf["model_version"] }}',
    'dataset_parameters': '{{ dag_run.conf["dataset_parameters"] }}',
    'model_config': '{{ dag_run.conf["model_config"] }}',
}


def upload_evaluation_results(**context):
    bucket = context['templates_dict']['evaluation_results_bucket']
    results_directory = context['templates_dict']['evaluation_results_dir']

    local_folder_to_s3(
        s3_conn_id='S3', s3_bucket=bucket,
        s3_path=PosixPath(''), local_path=SHARED_FOLDER / 'evaluation_results' / results_directory,
    )


def update_mlflow_description(**context):
    import mlflow
    import json
    from io import StringIO

    model_name = context['templates_dict']['model_name']
    model_version = context['templates_dict']['model_version']
    bucket = context['templates_dict']['evaluation_results_bucket']
    results_directory = context['templates_dict']['evaluation_results_dir']
    mlflow_url = f"{context['templates_dict']['s3_host']}:{context['templates_dict']['mlflow_port']}"
    s3_url = f"{context['templates_dict']['s3_host']}:{context['templates_dict']['s3_port']}"
    model_config = json.loads(context['templates_dict']['model_config'])

    def create_description():
        sio = StringIO()
        sio.write(f'![]({s3_url}/{bucket}/{results_directory}/test_results.png)\n')
        sio.write('<details>\n')
        sio.write('<summary>Model Parameters</summary>\n')
        sio.write('\n'.join(model_config))
        sio.write('\n</details>\n')
        return sio.getvalue()

    client = mlflow.tracking.MlflowClient(registry_uri=mlflow_url)
    client.update_model_version(
        name=model_name,
        version=model_version,
        description=create_description()
    )


def create_evaluate_model_dag(args, dag_id='evaluate-model'):
    args = {**DEFAULT_ARGS, **args}
    args['evaluation_results_dir'] = '{{ dict_hash({"model_name": ' + args["model_name"] + ', "model_version": ' + args["model_version"] + '}) }}'

    dag = DAG(
        dag_id,
        schedule_interval=None,
        catchup=False,
        default_args=args,
        user_defined_macros={
            'dict_hash': dict_hash,
            'get_aws_credentials': get_aws_credentials
        },
    )

    with dag:
        evaluation_operator = DockerOperatorExtended(
            task_id='run-simulation',
            dag=dag,
            image='alexdrydew/mpdroot',
            docker_url='unix://var/run/docker.sock',
            output_remote_mappings=[DockerOperatorRemoteMapping(
                remote_path=f'/{args["evaluation_results_dir"]}', mount_path='/evaluation_results',
                bucket=args["evaluation_results_bucket"],
            )],
            mounts=[
                Mount(source='/home/alexsuh/TPC/airflow/shared',
                      target=f'/shared', type='bind')
            ],
            environment={
                'ONNX_MODEL_NAME': args['model_name'],
                'ONNX_MODEL_VERSION': args['model_version'],
            },
            command=[
                'sh', '-c',
                'cd macro/mpd && root -l -b -q runMC.C && root -l -b -q reco.C && '
                f'curl -o /evaluation_results/test_results.png https://mlflow.org/images/MLflow-logo-final-white-TM.png'
            ],
        )

        update_description = PythonOperator(
            task_id='update-mlflow-description',
            dag=dag,
            python_callable=update_mlflow_description,
            templates_dict=args,
            provide_context=True,
        )

        evaluation_operator >> update_description

    return dag


dag = create_evaluate_model_dag(DEFAULT_ARGS)
