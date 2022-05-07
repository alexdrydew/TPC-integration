import uuid

import pytest
from pathlib import Path


@pytest.fixture(scope='session')
def s3_access_key_id():
    return str(uuid.uuid4())


@pytest.fixture(scope='session')
def s3_secret_access_key():
    return str(uuid.uuid4())


@pytest.fixture(scope='session')
def buckets_names():
    return {
        'airflow_bucket': str(uuid.uuid4()),
        'datasets_bucket': str(uuid.uuid4()),
        'saved_models_bucket': str(uuid.uuid4()),
        'evaluation_results_bucket': str(uuid.uuid4()),
        'tensorboard_bucket': str(uuid.uuid4())
    }


@pytest.fixture(scope='session')
def project_root():
    return Path(__file__).parent.parent
