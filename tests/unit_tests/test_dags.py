import pytest


@pytest.mark.parametrize(
    "dag_id",
    ["train-model", "get-dataset", "convert-and-upload-model", "evaluate-model"],
)
def test_load_dag(dag_id, init_test_airflow):
    # import after initialization
    from airflow.models import DagBag

    dagbag = DagBag(read_dags_from_db=True)
    dag = dagbag.get_dag(dag_id=dag_id)
    assert dagbag.import_errors == {}
    assert dag is not None
