FROM silikhon/tensorflow2:v1

RUN pip install tf2onnx==1.9.2 \
                 mlflow==1.23.1 \
                 onnxruntime==1.10.0 \
                 boto3==1.21.21 \
                 protobuf==3.19.4
# https://github.com/NVIDIA/nvidia-docker/issues/1624
RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list && apt-get clean
RUN apt-get update && apt-get install -y git

WORKDIR /
RUN git clone https://github.com/SiLiKhon/TPC-FastSim.git
WORKDIR TPC-FastSim
RUN git config user.name 'docker'
RUN git config user.email 'docker@example.com'
# model export support
RUN git reset --hard 7de367f
# custom logs folder support
RUN git cherry-pick 89d6d2e
RUN chmod -R 777 /TPC-FastSim
COPY /entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
