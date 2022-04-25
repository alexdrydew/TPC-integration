FROM silikhon/mpdroot-prereq:v2

WORKDIR /
RUN git clone https://github.com/alexdrydew/mpdroot
RUN curl -L -O https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-linux-x64-1.10.0.tgz && \
    mkdir -p /mpdroot/external/onnxruntime && \
    tar -xvzf onnxruntime-linux-x64-1.10.0.tgz -C mpdroot/external/onnxruntime --strip-components=1
RUN mkdir /mpdroot/build/

WORKDIR /mpdroot/build
RUN source scl_source enable devtoolset-7 && . ../SetEnv.sh && cmake .. && make -j `nproc`
WORKDIR /mpdroot
COPY /entrypoint.sh ./entrypoint.sh
COPY /create_dataset.sh ./create_dataset.sh
RUN chmod -R 777 .
RUN chmod +x ./entrypoint.sh
ENV DISPLAY localhost:0
ENTRYPOINT ["./entrypoint.sh"]
