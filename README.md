```bash
sudo useradd airflow
```

Run tests:
```bash
pip install -r test-requirements
# generate python client for airflow 2.3.0 using OpenAPI
git clone git@github.com:apache/airflow-client-python.git
cd airflow-client-python
rm -r airflow_client/*
git clone git@github.com:apache/airflow
cd airflow/clients
git checkout tags/2.3.0
# use patched specification
cp ../../../airflow/v1.yaml ../airflow/api_connexion/openapi/v1.yaml
python ../../../add_auth_to_openapi.py --config ../airflow/api_connexion/openapi/v1.yaml --auth_options Basic
./gen/python.sh ../airflow/api_connexion/openapi/v1.yaml ../../airflow_client/
cd ../../..
pip install attrs==20.3.0
pip install cattrs==1.1.1
pip install ./airflow-client-python --no-cache --force
rm -r airflow-client-python

pytest tests/unit_tests
pytest tests/functional_tests
```