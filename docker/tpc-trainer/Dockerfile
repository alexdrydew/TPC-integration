FROM silikhon/tensorflow2:v1
RUN pip install tf2onnx==1.9.2 \
                 mlflow==1.23.1 \
                 onnxruntime==1.10.0 \
                 boto3==1.21.21 \
                 protobuf==3.19.4
RUN apt-get update && apt-get install -y git

WORKDIR /
RUN git clone https://github.com/SiLiKhon/TPC-FastSim.git
WORKDIR TPC-FastSim
RUN git reset --hard 8f0eef0
RUN chmod -R 777 .
ENTRYPOINT ["python"]
