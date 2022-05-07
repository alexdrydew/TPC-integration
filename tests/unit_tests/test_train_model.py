import pytest
import yaml
from io import BytesIO

from pathlib import Path


@pytest.fixture
def model_config():
    return {
        "param1": 1,
        "param2": {"param3": 4},
        "param4": ["some-string-1", "some-string-2"],
    }


def test_create_config(init_test_airflow, model_config, buckets, mocked_s3_client):
    from train_model import create_config

    airflow_bucket = buckets["airflow_bucket"]

    create_config(model_config, "model_dir", airflow_bucket)

    f = BytesIO()
    mocked_s3_client.download_fileobj(
        Fileobj=f, Bucket=airflow_bucket, Key="tmp/model_dir/models/configs/model.yaml"
    )
    loaded = yaml.safe_load(f.getvalue().decode("utf-8"))
    assert loaded == model_config


@pytest.fixture
def saved_model_local_files():
    return list(
        (Path(__file__).parent / "blob" / "saved_model_dir").glob("**/*.*")
    )


@pytest.fixture
def s3_saved_model_intermediate(mocked_s3_client, buckets, saved_model_local_files):
    for file in saved_model_local_files:
        mocked_s3_client.upload_file(
            str(file),
            Bucket=buckets["saved_models_bucket"],
            Key=f'intermediate/{file.relative_to(Path(__file__).parent / "blob")}',
        )
    return "saved_model_dir"


def test_copy_model_to_done_folder(
    init_test_airflow,
    s3_saved_model_intermediate,
    mocked_s3_resource,
    buckets,
    saved_model_local_files,
):
    from train_model import copy_model_to_done_folder

    copy_model_to_done_folder(
        s3_saved_model_intermediate,
        buckets["saved_models_bucket"],
        {"dataset_param_1": "1"},
    )

    bucket = mocked_s3_resource.Bucket(buckets["saved_models_bucket"])
    s3_files = list(
        bucket.objects.filter(Prefix=f"done/{s3_saved_model_intermediate}/")
    )

    uploaded = set(file.key[len("done/") :] for file in s3_files)
    local = set(
        str(file.relative_to(Path(__file__).parent / "blob"))
        for file in saved_model_local_files
    )
    assert f"{s3_saved_model_intermediate}/dataset.yaml" in uploaded
    uploaded.remove(f"{s3_saved_model_intermediate}/dataset.yaml")

    assert uploaded == local


def test_decide_if_continue(init_test_airflow, s3_saved_model_intermediate, buckets):
    from train_model import decide_if_continue

    assert (
        decide_if_continue(
            True, s3_saved_model_intermediate, buckets["saved_models_bucket"]
        )
        == "prepare-config"
    )
    assert (
        decide_if_continue(
            False, s3_saved_model_intermediate, buckets["saved_models_bucket"]
        )
        == "train-model-continue"
    )
    assert (
        decide_if_continue(True, "fake-path", buckets["saved_models_bucket"])
        == "prepare-config"
    )
    assert (
        decide_if_continue(False, "fake-path", buckets["saved_models_bucket"])
        == "prepare-config"
    )
