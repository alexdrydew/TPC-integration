import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from docker.types import Mount, DeviceRequest
from pathlib import Path
from get_dataset import create_get_dataset_operators
from util import DockerOperatorExtended, dict_hash, DockerOperatorRemoteMapping, COMMON_DEFAULT_ARGS, \
    get_aws_credentials

DEFAULT_ARGS = {
    **COMMON_DEFAULT_ARGS,
    'start_date': pendulum.now(),
    'model_name': '{{ dag_run.conf["model_name"] }}',
    'model_config': '{{ dag_run.conf["model_config"] }}',
    'dataset_parameters': '{{ dag_run.conf["dataset_parameters"] }}',
    'ignore_saves': '{{ dag_run.conf.get("ignore_saves", False)}}',
}


def create_config(**context):
    import yaml
    import tempfile
    from util import local_folder_to_s3

    model_config = context['templates_dict']['model_config']
    model_dir = context['templates_dict']['model_dir']

    with tempfile.TemporaryDirectory() as file_dir:
        file_dir = Path(file_dir)
        (file_dir / 'models' / 'configs').mkdir(exist_ok=True, parents=True)
        with open(file_dir / 'models' / 'configs' / 'model.yaml', 'w') as f:
            yaml.dump(model_config, f, default_flow_style=False)

        local_folder_to_s3('S3', 'airflow', s3_path=f'/tmp/{model_dir}', local_path=file_dir / 'models')


def copy_model_to_done_folder(**context):
    bucket = context['templates_dict']['saved_models_bucket']
    model_dir = context['templates_dict']['model_dir']

    s3_hook = S3Hook(aws_conn_id='S3')
    for key in s3_hook.list_keys(bucket_name=bucket, prefix=f'intermediate/{model_dir}'):
        s3_hook.copy_object(
            source_bucket_name=bucket, dest_bucket_name=bucket,
            source_bucket_key=key, dest_bucket_key=key.replace('intermediate', 'done'),
        )


def decide_if_continue(**context):
    s3_hook = S3Hook(aws_conn_id='S3')
    ignore_saves = context['templates_dict']['ignore_saves']
    model_dir = context['templates_dict']['model_dir']
    bucket = context['templates_dict']['saved_models_bucket']

    if ignore_saves or not s3_hook.check_for_prefix(bucket_name=bucket, prefix=f'intermediate/{model_dir}', delimiter='/'):
        return 'prepare-config'
    return 'train-model-continue',


def create_train_model_dag(args, dag_id='train-model'):
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

        get_dataset_operator = create_get_dataset_operators(dag, args)

        prepare_config_operator = PythonOperator(
            task_id='prepare-config',
            dag=dag,
            python_callable=create_config,
            templates_dict=args,
            provide_context=True,
        )

        decide_if_continue_operator = BranchPythonOperator(
            task_id='decide-if-continue',
            dag=dag,
            python_callable=decide_if_continue,
            templates_dict=args,
            provide_context=True,
        )

        common_train_args = {
            'dag': dag,
            'image': 'alexdrydew/tpc-trainer',
            'docker_url': 'unix://var/run/docker.sock',
            'device_requests': [DeviceRequest(capabilities=[['gpu']], count=1)],
            'output_remote_mappings': [
                DockerOperatorRemoteMapping(
                    bucket=args["saved_models_bucket"],
                    remote_path=f'/intermediate/{args["model_dir"]}',
                    mount_path='/TPC-FastSim/saved_models',
                )
            ],
            'map_output_on_fail': True
        }

        train_model_operator_from_start = DockerOperatorExtended(
            task_id='train-model-from-start',
            **common_train_args,
            input_remote_mappings=[
                DockerOperatorRemoteMapping(
                    bucket='airflow',
                    remote_path=f'/tmp/{args["model_dir"]}/models/configs',
                    mount_path='/TPC-FastSim/models/configs',
                ),
                DockerOperatorRemoteMapping(
                    bucket='datasets',
                    remote_path=f'/{args["dataset_dir"]}',
                    mount_path='/TPC-FastSim/data/data_v4',
                ),
            ],
            command=[
                'run_model_v4.py',
                '--checkpoint_name', 'model', '--config', 'models/configs/model.yaml',
            ],
        )
        train_model_operator_continue = DockerOperatorExtended(
            task_id='train-model-continue',
            **common_train_args,
            input_remote_mappings=[
                DockerOperatorRemoteMapping(
                    bucket='airflow',
                    remote_path=f'/tmp/{args["model_dir"]}/models/configs',
                    mount_path='/TPC-FastSim/models/configs',
                ),
                DockerOperatorRemoteMapping(
                    bucket=f'{args["datasets_bucket"]}',
                    remote_path=f'/{args["dataset_dir"]}',
                    mount_path='/TPC-FastSim/data/data_v4',
                ),
                DockerOperatorRemoteMapping(
                    bucket=args["saved_models_bucket"],
                    remote_path=f'/intermediate/{args["model_dir"]}',
                    mount_path='/TPC-FastSim/saved_models',
                ),
            ],
            command=[
                'run_model_v4.py',
                '--checkpoint_name', 'model',
            ],
        )

        copy_trained_model_operator = PythonOperator(
            task_id='copy-to-done',
            dag=dag,
            python_callable=copy_model_to_done_folder,
            templates_dict=args,
            provide_context=True,
            trigger_rule='one_success',
        )

        args.pop('start_date')
        trigger_convert_model_dag = TriggerDagRunOperator(
            task_id='trigger-convert-model',
            dag=dag,
            trigger_dag_id='convert-and-upload-model',
            conf=args
        )

        get_dataset_operator >> decide_if_continue_operator

        decide_if_continue_operator >> prepare_config_operator >> train_model_operator_from_start >> copy_trained_model_operator
        decide_if_continue_operator >> train_model_operator_continue >> copy_trained_model_operator
        copy_trained_model_operator >> trigger_convert_model_dag

    return dag


dag = create_train_model_dag(DEFAULT_ARGS)
