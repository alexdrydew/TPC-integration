import time

import pytest
from airflow_client.client.api import dag_api, dag_run_api, task_instance_api
from airflow_client.client.model.dag_run import DAGRun
from airflow_client.client.model.dag_state import DagState
from airflow_client.client.model.task_state import TaskState


def run_dag_until_complete(airflow_client_instance, dag_id, conf):
    api_instance = dag_run_api.DAGRunApi(airflow_client_instance)
    dag_run = api_instance.post_dag_run(dag_id, DAGRun(conf=conf))
    dag_run = api_instance.get_dag_run(
        dag_id=dag_id, dag_run_id=dag_run.dag_run_id
    )
    while dag_run.state == DagState('running') or dag_run.state == DagState('queued'):
        time.sleep(1)
        dag_run = api_instance.get_dag_run(
            dag_id=dag_id, dag_run_id=dag_run.dag_run_id
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


def test_train_model(
    run_docker_compose, airflow_client_instance, generated_dataset_params, model_config
):
    dag_run = run_dag_until_complete(
        airflow_client_instance, 'train-model',
        {
            'model_name': 'test',
            'model_config': model_config,
            'dataset_parameters': generated_dataset_params,
        }
    )
    api_instance = task_instance_api.TaskInstanceApi(airflow_client_instance)
    task_instances_collection = api_instance.get_task_instances(
        dag_id='train-model', dag_run_id=dag_run.dag_run_id,
    )