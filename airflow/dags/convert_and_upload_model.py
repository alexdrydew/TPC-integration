import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from util import DockerOperatorExtended, DockerOperatorRemoteMapping, get_aws_credentials, COMMON_DEFAULT_ARGS, \
    dict_hash

DEFAULT_ARGS = {
    **COMMON_DEFAULT_ARGS,
    'start_date': pendulum.now(),
    'model_name': '{{ dag_run.conf["model_name"] }}',
}


def get_last_version(**context):
    import mlflow

    model_name = context['templates_dict']['model_name']
    host = context['templates_dict']['s3_host']
    port = context['templates_dict']['mlflow_port']
    client = mlflow.tracking.MlflowClient(tracking_uri=f'{host}:{port}')
    new_version = client.get_latest_versions(model_name)[0].version

    return new_version


def create_convert_and_upload_model_dag(args, dag_id='convert-and-upload-model'):
    args = {**DEFAULT_ARGS, **args}

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
        convert_operator = DockerOperatorExtended(
            task_id='convert-and-upload',
            dag=dag,
            image='alexdrydew/tpc-trainer:latest',
            docker_url='unix://var/run/docker.sock',
            network_mode='host',
            input_remote_mappings=[
                DockerOperatorRemoteMapping(
                    bucket=args['saved_models_bucket'], remote_path=f'/done/{args["model_dir"]}',
                    mount_path='/TPC-FastSim/saved_models'
                )
            ],
            command=[
                'export_model.py',
                '--checkpoint_name', 'model',
                '--export_format', 'onnx',
                '--upload_to_mlflow',
                '--aws_access_key_id', args['s3_access_key'],
                '--aws_secret_access_key', args['s3_secret_access_key'],
                '--mlflow_url', args['s3_host'],
                '--mlflow_s3_port', args['s3_port'],
                '--mlflow_tracking_port', args['mlflow_port'],
                '--mlflow_model_name', args['model_name'],
            ],
        )

        get_version_operator = PythonOperator(
            task_id='get-last-version',
            dag=dag,
            python_callable=get_last_version,
            templates_dict=args,
            provide_context=True,
        )

        args.pop('start_date')
        args['model_version'] = '{{ task_instance.xcom_pull("get-last-version") }}'
        trigger_evaluation_operator = TriggerDagRunOperator(
            task_id='evaluate-model',
            dag=dag,
            trigger_dag_id='evaluate-model',
            conf=args
        )

        convert_operator >> get_version_operator >> trigger_evaluation_operator

    return dag


dag = create_convert_and_upload_model_dag(DEFAULT_ARGS)
