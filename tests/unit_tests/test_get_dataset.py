import pytest
from pathlib import Path


@pytest.fixture
def mocked_dataset_dir(mocked_s3_client, buckets):
    dataset_dir_name = "test_dataset_dir"
    dataset_path = f"{dataset_dir_name}/raw/digits.dat"
    mocked_s3_client.upload_file(
        str(Path(__file__).parent / "blob" / "digits.dat"),
        Bucket=buckets["datasets_bucket"],
        Key=dataset_path,
    )
    return dataset_dir_name


def test_dataset_exists(init_test_airflow, mocked_dataset_dir, buckets):
    from get_dataset import dataset_exists

    assert (
        dataset_exists(mocked_dataset_dir, buckets["datasets_bucket"]) == "do-nothing"
    )
    assert (
        dataset_exists("fake-dataset-path", buckets["datasets_bucket"])
        == "create-dataset"
    )
