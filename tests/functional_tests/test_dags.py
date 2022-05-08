import os
import time

import boto3
import mlflow
import numpy as np
import onnxruntime
import pytest
import requests
from airflow_client.client.api import dag_api, dag_run_api, task_instance_api
from airflow_client.client.model.dag_run import DAGRun
from airflow_client.client.model.dag_state import DagState
from airflow_client.client.model.task_state import TaskState


def run_dag_until_complete(airflow_client_instance, dag_id, conf):
    api_instance = dag_run_api.DAGRunApi(airflow_client_instance)
    dag_run = api_instance.post_dag_run(dag_id, DAGRun(conf=conf))
    dag_run = wait_dag_run_until_complete(api_instance, dag_run)
    return dag_run


def wait_dag_run_until_complete(dag_run_api_instance, dag_run):
    dag_run = dag_run_api_instance.get_dag_run(
        dag_id=dag_run.dag_id, dag_run_id=dag_run.dag_run_id
    )
    while dag_run.state == DagState('running') or dag_run.state == DagState('queued'):
        time.sleep(1)
        dag_run = dag_run_api_instance.get_dag_run(
            dag_id=dag_run.dag_id, dag_run_id=dag_run.dag_run_id
        )
    return dag_run


@pytest.fixture(scope='session')
def generated_dataset_params():
    return {'size': 100}


@pytest.fixture(scope='session')
def generated_dataset(
    run_docker_compose, airflow_client_instance, generated_dataset_params
):
    dag_run = run_dag_until_complete(
        airflow_client_instance,
        'get-dataset',
        {'dataset_parameters': generated_dataset_params},
    )
    assert dag_run.state == DagState('success')


def test_dags_initialized(run_docker_compose, airflow_client_instance):
    api_instance = dag_api.DAGApi(airflow_client_instance)
    dags = api_instance.get_dags().dags
    assert set(dag.dag_id for dag in dags) == {
        'get-dataset',
        'train-model',
        'evaluate-model',
        'convert-and-upload-model',
    }


def test_get_dataset_skip_generation(
    run_docker_compose,
    airflow_client_instance,
    generated_dataset,
    generated_dataset_params,
):
    dag_run = run_dag_until_complete(
        airflow_client_instance, 'get-dataset', {'dataset_parameters': generated_dataset_params},
    )
    api_instance = task_instance_api.TaskInstanceApi(airflow_client_instance)

    task_instances_collection = api_instance.get_task_instances(
        dag_id='get-dataset', dag_run_id=dag_run.dag_run_id,
    )

    expected_states = {
        'check-dataset-exists': 'success',
        'create-dataset': 'skipped',
        'convert-dataset': 'skipped',
        'dataset-ready': 'success',
        'do-nothing': 'success',
    }

    for task_instance in task_instances_collection.task_instances:
        assert TaskState(expected_states[task_instance.task_id]) == task_instance.state,\
            f'Task {task_instance.task_id} is in state {task_instance.state}'


@pytest.fixture(scope='session')
def model_config():
    return {
        'latent_dim': 32,
        'batch_size': 32,
        'lr': 0.0001,
        'lr_schedule_rate': 0.999,
        'num_disc_updates': 8,
        'gp_lambda': 10,
        'gpdata_lambda': 0,
        'cramer': False,
        'stochastic_stepping': True,
        'save_every': 1,
        'num_epochs': 2,
        'feature_noise_power': None,
        'feature_noise_decay': None,
        'data_version': 'data_v4',
        'pad_range': [-3, 5],
        'time_range': [-7, 9],
        'scaler': 'logarithmic',
        'architecture': {
            'generator': [
                {
                    'block_type': 'fully_connected',
                    'arguments': {
                        'units': [32, 64, 64, 64, 128],
                        'activations': [
                            'elu',
                            'elu',
                            'elu',
                            'elu',
                            ' ( lambda x, shift=0.01, val=np.log10(2), v0=np.log10(2) / 10: ( tf.where( x > shift, val + x - shift, v0 + tf.keras.activations.elu( x, alpha=(v0 * shift / (val - v0)) ) * (val - v0) / shift ) ) )',
                        ],
                        'kernel_init': 'glorot_uniform',
                        'input_shape': [37],
                        'output_shape': [8, 16],
                        'name': 'generator',
                    },
                }
            ],
            'discriminator': [
                {
                    'block_type': 'connect',
                    'arguments': {
                        'vector_shape': [5],
                        'img_shape': [8, 16],
                        'vector_bypass': False,
                        'concat_outputs': True,
                        'name': 'discriminator_tail',
                        'block': {
                            'block_type': 'conv',
                            'arguments': {
                                'filters': [16, 16, 32, 32, 64, 64],
                                'kernel_sizes': [3, 3, 3, 3, 3, 2],
                                'paddings': [
                                    'same',
                                    'same',
                                    'same',
                                    'same',
                                    'valid',
                                    'valid',
                                ],
                                'activations': [
                                    'elu',
                                    'elu',
                                    'elu',
                                    'elu',
                                    'elu',
                                    'elu',
                                ],
                                'poolings': [None, [1, 2], None, 2, None, None],
                                'kernel_init': 'glorot_uniform',
                                'input_shape': None,
                                'output_shape': [64],
                                'dropouts': [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                                'name': 'discriminator_conv_block',
                            },
                        },
                    },
                },
                {
                    'block_type': 'fully_connected',
                    'arguments': {
                        'units': [128, 1],
                        'activations': ['elu', None],
                        'kernel_init': 'glorot_uniform',
                        'input_shape': [69],
                        'output_shape': None,
                        'name': 'discriminator_head',
                    },
                },
            ],
        },
    }


@pytest.fixture(scope='session')
def trained_model(run_docker_compose, airflow_client_instance, generated_dataset_params, model_config):
    dag_run = run_dag_until_complete(
        airflow_client_instance, 'train-model',
        {
            'model_name': 'test',
            'model_config': model_config,
            'dataset_parameters': generated_dataset_params,
            'ignore_saves': False,
        }
    )
    return dag_run


def test_trained_model_successful(airflow_client_instance, trained_model, s3_access_key_id, s3_secret_access_key, hostname, ports, buckets_names):
    api_instance = task_instance_api.TaskInstanceApi(airflow_client_instance)
    task_instances_collection = api_instance.get_task_instances(
        dag_id='train-model', dag_run_id=trained_model.dag_run_id,
    )
    expected_states = {
        'prepare-config': 'success',
        'decide-if-continue': 'success',
        'train-model-from-start': 'success',
        'train-model-continue': 'skipped',
        'copy-to-done': 'success',
        'trigger-convert-model': 'success',
    }

    for task_instance in task_instances_collection.task_instances:
        if task_instance.task_id in expected_states:
            assert TaskState(expected_states[task_instance.task_id]) == task_instance.state, \
                f'Task {task_instance.task_id} is in state {task_instance.state}'

    trigger_task = api_instance.get_task_instance(
        dag_id='train-model', dag_run_id=trained_model.dag_run_id, task_id='trigger-convert-model'
    )
    model_dir = trigger_task.rendered_fields['conf']['saved_model_dir']

    check_model_dir(buckets_names, hostname, model_dir, ports, s3_access_key_id, s3_secret_access_key)


def check_model_dir(buckets_names, hostname, model_dir, ports, s3_access_key_id, s3_secret_access_key):
    s3_resource = boto3.resource(
        service_name='s3',
        aws_access_key_id=s3_access_key_id,
        aws_secret_access_key=s3_secret_access_key,
        endpoint_url=f'http://{hostname}:{ports["MINIO_PORT"]}',
    )
    bucket = s3_resource.Bucket(buckets_names['saved_models_bucket'])
    s3_files = list(bucket.objects.filter(Prefix=f"done/{model_dir}/"))
    assert s3_files


def test_train_model_continued(
    airflow_client_instance, generated_dataset_params, model_config, trained_model, buckets_names, hostname, ports, s3_access_key_id, s3_secret_access_key
):
    dag_run = run_dag_until_complete(
        airflow_client_instance, 'train-model',
        {
            'model_name': 'test',
            'model_config': model_config,
            'dataset_parameters': generated_dataset_params,
            'ignore_saves': False,
        }
    )
    api_instance = task_instance_api.TaskInstanceApi(airflow_client_instance)
    task_instances_collection = api_instance.get_task_instances(
        dag_id='train-model', dag_run_id=dag_run.dag_run_id,
    )
    expected_states = {
        'prepare-config': 'skipped',
        'decide-if-continue': 'success',
        'train-model-from-start': 'skipped',
        'train-model-continue': 'success',
        'copy-to-done': 'success',
        'trigger-convert-model': 'success',
    }

    for task_instance in task_instances_collection.task_instances:
        if task_instance.task_id in expected_states:
            assert TaskState(expected_states[task_instance.task_id]) == task_instance.state, \
                f'Task {task_instance.task_id} is in state {task_instance.state}'

    trigger_task = api_instance.get_task_instance(
        dag_id='train-model', dag_run_id=dag_run.dag_run_id, task_id='trigger-convert-model'
    )
    model_dir = trigger_task.rendered_fields['conf']['saved_model_dir']
    check_model_dir(buckets_names, hostname, model_dir, ports, s3_access_key_id, s3_secret_access_key)


@pytest.fixture(scope='session')
def triggered_convert_model(airflow_client_instance, trained_model):
    api_instance = dag_run_api.DAGRunApi(airflow_client_instance)
    convert_dag_run = api_instance.get_dag_run('convert-and-upload-model', dag_run_id=trained_model.dag_run_id)
    convert_dag_run = wait_dag_run_until_complete(api_instance, convert_dag_run)
    return convert_dag_run


def test_triggered_convert_model_successful(airflow_client_instance, triggered_convert_model):
    assert triggered_convert_model.state == DagState('success')
    task_api_instance = task_instance_api.TaskInstanceApi(airflow_client_instance)
    task_instances_collection = task_api_instance.get_task_instances(
        dag_id='convert-and-upload-model', dag_run_id=triggered_convert_model.dag_run_id
    )
    for task_instance in task_instances_collection.task_instances:
        assert task_instance.state == TaskState('success')


@pytest.fixture(scope='session')
def set_mlflow_s3_credentials(s3_access_key_id, s3_secret_access_key, hostname, ports):
    os.environ['AWS_ACCESS_KEY_ID'] = s3_access_key_id
    os.environ['AWS_SECRET_ACCESS_KEY'] = s3_secret_access_key
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = f'http://{hostname}:{ports["MINIO_PORT"]}'


def test_model_is_uploaded_to_mlflow(triggered_convert_model, airflow_client_instance, hostname, ports, set_mlflow_s3_credentials):
    task_api_instance = task_instance_api.TaskInstanceApi(airflow_client_instance)
    trigger_evaluation_task = task_api_instance.get_task_instance(
        dag_id=triggered_convert_model.dag_id, dag_run_id=triggered_convert_model.dag_run_id,
        task_id='trigger-evaluate-model'
    )

    conf = trigger_evaluation_task.rendered_fields['conf']
    model_version, model_name = conf['model_version'], conf['model_name']

    client = mlflow.tracking.MlflowClient(tracking_uri=f'http://{hostname}:{ports["MLFLOW_PORT"]}')
    run_id = client.get_model_version(model_name, model_version).run_id
    artifacts = client.list_artifacts(run_id, 'model_onnx')
    assert len(artifacts) == 6


@pytest.fixture(scope='session')
def triggered_evaluate_model(airflow_client_instance, triggered_convert_model):
    api_instance = dag_run_api.DAGRunApi(airflow_client_instance)
    evaluate_dag_run = api_instance.get_dag_run('evaluate-model', dag_run_id=triggered_convert_model.dag_run_id)
    evaluate_dag_run = wait_dag_run_until_complete(api_instance, evaluate_dag_run)
    return evaluate_dag_run


def test_triggered_evaluate_model_successful(airflow_client_instance, triggered_evaluate_model, hostname, ports, set_mlflow_s3_credentials):
    assert triggered_evaluate_model.state == DagState('success')
    task_api_instance = task_instance_api.TaskInstanceApi(airflow_client_instance)
    task_instances_collection = task_api_instance.get_task_instances(
        dag_id='evaluate-model', dag_run_id=triggered_evaluate_model.dag_run_id
    )
    for task_instance in task_instances_collection.task_instances:
        assert task_instance.state == TaskState('success')

    client = mlflow.tracking.MlflowClient(tracking_uri=f'http://{hostname}:{ports["MLFLOW_PORT"]}')

    model_version, model_name = int(triggered_evaluate_model.conf['model_version']), triggered_evaluate_model.conf['model_name']
    assert len(client.get_model_version(model_name, model_version).description) > 0


def test_result_model_raw_requested(airflow_client_instance, triggered_evaluate_model, hostname, ports, set_mlflow_s3_credentials):
    model_name = triggered_evaluate_model.conf['model_name']

    model_version = int(triggered_evaluate_model.conf['model_version'])

    download_uri = requests.get(
        f'http://{hostname}:{ports["MLFLOW_PORT"]}/api/2.0/preview/mlflow/model-versions/get-download-uri?name={model_name}&version={model_version}'
    ).json()['artifact_uri']
    s3_path = (download_uri + '/model.onnx').replace('s3://', '/')

    onnx_model = requests.get(f'http://{hostname}:{ports["MINIO_PORT"]}' + s3_path).content
    session = onnxruntime.InferenceSession(onnx_model)
    session.run(None, {'x': np.ones((1, 4), dtype=np.float32)})
