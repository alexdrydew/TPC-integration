import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import boto3
import pytest
from moto.moto_server.threaded_moto_server import ThreadedMotoServer


@pytest.fixture(scope="session")
def test_airflow_root():
    return Path(__file__).parent / "airflow"


@pytest.fixture(scope="session")
def production_airflow_root():
    return Path(__file__).parent.parent / "airflow"


@pytest.fixture(scope="session")
def s3_host():
    return "127.0.0.1"


@pytest.fixture(scope="session")
def s3_port(s3_host):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((s3_host, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture(scope="session")
def add_dags_dir_to_path(production_airflow_root):
    sys.path.insert(0, str(production_airflow_root / "dags"))


@pytest.fixture(scope="session")
def init_test_airflow(
    airflow_variables, test_airflow_root, add_dags_dir_to_path, production_airflow_root
):
    shutil.rmtree(str(test_airflow_root), ignore_errors=True)
    test_airflow_root.mkdir()
    (test_airflow_root / "dags").symlink_to(
        production_airflow_root / "dags", target_is_directory=True
    )
    subprocess.run(["airflow", "db", "reset", "--yes"])
    yield
    shutil.rmtree(str(test_airflow_root))


@pytest.fixture
def s3_server(s3_host, s3_port):
    server = ThreadedMotoServer(ip_address=s3_host, port=s3_port)
    server.start()
    yield
    server.stop()


@pytest.fixture
def mocked_s3_client(
    s3_server, s3_host, s3_port, s3_access_key_id, s3_secret_access_key
):
    client = boto3.client(
        service_name="s3",
        aws_access_key_id=s3_access_key_id,
        aws_secret_access_key=s3_secret_access_key,
        endpoint_url=f"http://{s3_host}:{s3_port}",
    )
    return client


@pytest.fixture
def mocked_s3_resource(
    s3_server, s3_host, s3_port, s3_access_key_id, s3_secret_access_key
):
    client = boto3.resource(
        service_name="s3",
        aws_access_key_id=s3_access_key_id,
        aws_secret_access_key=s3_secret_access_key,
        endpoint_url=f"http://{s3_host}:{s3_port}",
    )
    return client


@pytest.fixture
def buckets(mocked_s3_client, buckets_names):
    for bucket in buckets_names.values():
        mocked_s3_client.create_bucket(Bucket=bucket)
    return buckets_names


@pytest.fixture(scope="session")
def airflow_variables(
    test_airflow_root,
    s3_host,
    s3_port,
    s3_access_key_id,
    s3_secret_access_key,
    buckets_names,
):
    os.environ["AIRFLOW_VAR_AIRFLOW_BUCKET"] = buckets_names["airflow_bucket"]
    os.environ["AIRFLOW_VAR_DATASETS_BUCKET"] = buckets_names["datasets_bucket"]
    os.environ["AIRFLOW_VAR_SAVED_MODELS_BUCKET"] = buckets_names["saved_models_bucket"]
    os.environ["AIRFLOW_VAR_EVALUATION_RESULTS_BUCKET"] = buckets_names[
        "evaluation_results_bucket"
    ]
    os.environ["AIRFLOW_VAR_MLFLOW_HOST"] = "tracking-server"
    os.environ["AIRFLOW_VAR_MLFLOW_TRACKING_PORT"] = "5000"
    os.environ["AIRFLOW_VAR_TENSORBOARD_BUCKET"] = "tensorboard"
    os.environ["AIRFLOW_VAR_REVERSE_PROXY_HOST"] = s3_host
    os.environ["AIRFLOW_VAR_REVERSE_PROXY_S3_PORT"] = str(s3_port)
    os.environ[
        "AIRFLOW_CONN_S3"
    ] = f"s3:///?aws_access_key_id={s3_access_key_id}&aws_secret_access_key={s3_secret_access_key}&host=http://{s3_host}:{s3_port}"
    os.environ["AIRFLOW__CORE__LOAD_DEFAULT_CONNECTIONS"] = "False"
    os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
    os.environ["AIRFLOW__CORE__UNIT_TEST_MODE"] = "True"
    os.environ["AIRFLOW_HOME"] = str(test_airflow_root)