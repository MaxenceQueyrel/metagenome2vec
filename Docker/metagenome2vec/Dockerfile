#Distribution name
FROM maxence27/deep_learning:cuda113-cudnn8

ARG DEBIAN_FRONTEND="noninteractive"

RUN apt update -y && \
  apt install default-jdk -y

RUN pip install pyspark==3.2.1 && \
  pip install h2o==3.36.0.4 && \
  pip install h2o_pysparkling_3.2==3.36.0.4-1 && \
  pip install dill==0.3.4 && \
  pip install biopython==1.79 && \
  pip install pyarrow==6.0.1 && \
  pip install fasttext==0.9.2 && \
  pip install pytorch-lightning==1.5.5 && \
  pip install mlflow==1.22.0 && \
  pip install hydra-core==1.1.1

