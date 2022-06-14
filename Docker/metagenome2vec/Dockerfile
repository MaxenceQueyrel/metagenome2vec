#Distribution name
FROM maxence27/deep_learning:cuda114-cudnn8

ARG DEBIAN_FRONTEND="noninteractive"

RUN apt update -y && \
  apt install default-jdk -y

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

