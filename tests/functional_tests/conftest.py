import os
import socket
import uuid
from grp import getgrnam
from io import StringIO
from pathlib import Path
from pwd import getpwnam
from time import sleep

import pytest
from airflow_client.client.api.monitoring_api import MonitoringApi
from compose.cli.command import get_project
from compose.service import ImageType
import airflow_client.client


@pytest.fixture(scope='session')
def airflow_user():
    return str(uuid.uuid4())


@pytest.fixture(scope='session')
def airflow_password():
    return str(uuid.uuid4())


@pytest.fixture(scope='session')
def hostname():
    return 'localhost'


@pytest.fixture(scope='session')
def ports(hostname):
    ports_dict = {
        'AIRFLOW_PORT': None,
        'MLFLOW_PORT': None,
        'MINIO_PORT': None,
        'TENSORBOARD_PORT': None,
    }

    last_port = 1000
    for port_name in ports_dict:
        is_free = False
        while not is_free:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind((hostname, last_port))
                is_free = True
                s.close()
            except OSError:
                last_port += 1
        ports_dict[port_name] = last_port
        last_port += 1

    return ports_dict


@pytest.fixture(scope='session')
def initialize_env_file(
    buckets_names, s3_access_key_id, s3_secret_access_key, airflow_user,
    airflow_password, hostname, project_root, ports,
):
    sio = StringIO()

    # initialize bucket names
    for bucket, name in buckets_names.items():
        sio.write(f'{bucket.upper()}={name}\n')

    for port_name, port in ports.items():
        sio.write(f'{port_name}={port}\n')

    # initialize minio
    sio.write(f'MINIO_ACCESS_KEY={s3_access_key_id}\n')
    sio.write(f'MINIO_SECRET_ACCESS_KEY={s3_secret_access_key}\n')

    # initialize postgres
    sio.write(f'PG_USER={uuid.uuid4()}\n')
    sio.write(f'PG_PASSWORD={uuid.uuid4()}\n')

    # initialize airflow
    sio.write(f'AIRFLOW_USER={airflow_user}\n')
    sio.write(f'AIRFLOW_PASSWORD={airflow_password}\n')
    sio.write(f'AIRFLOW_DAGS={project_root / "airflow" / "dags"}\n')

    # nginx
    sio.write(f'REVERSE_PROXY_HOST={hostname}\n')
    sio.write(f'NGINX_CONF={project_root / "airflow" / "nginx.conf"}\n')

    # users
    sio.write(f'AIRFLOW_UID={getpwnam("airflow").pw_uid}\n')
    sio.write(f'DOCKER_GID={getgrnam("docker").gr_gid}\n')

    # dockerfiles
    sio.write(f'AIRFLOW_DOCKERFILE={project_root / "docker" / "airflow"}\n')
    sio.write(f'MLFLOW_DOCKERFILE={project_root / "docker" / "mlflow"}\n')
    sio.write(f'TENSORBOARD_DOCKERFILE={project_root / "docker" / "tensorboard"}\n')

    # docker compose
    sio.write(f'NETWORK_NAME={uuid.uuid4()}')

    env_filepath = Path(__file__).parent / '.env'
    with open(env_filepath, 'w') as f:
        f.write(sio.getvalue())
    yield env_filepath


@pytest.fixture(scope='session')
def run_docker_compose(initialize_env_file, hostname):
    project = get_project(project_dir=str(Path(__file__).parent))
    project.up()
    yield
    project.down(ImageType.none, include_volumes=True, remove_orphans=True)
    os.remove(initialize_env_file)


@pytest.fixture(scope='session')
def airflow_client_instance(hostname, airflow_user, airflow_password, ports):
    configuration = airflow_client.client.Configuration(
        host=f'http://{hostname}:{ports["AIRFLOW_PORT"]}/api/v1',
        username=airflow_user, password=airflow_password,
    )

    with airflow_client.client.ApiClient(configuration) as api_client:
        monitoring_api = MonitoringApi(api_client)

        def wait_for_service(retries=10):
            try:
                monitoring_api.get_health()
            except:
                if retries > 0:
                    sleep(5)
                    wait_for_service(retries - 1)
                else:
                    raise

        wait_for_service()
        yield api_client
