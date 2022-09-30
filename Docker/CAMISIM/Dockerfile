# Distribution name
FROM ubuntu:18.04

MAINTAINER Queyrel Maxence <maxencequeyrel@hotmail.com>

ENV DEBIAN_FRONTEND="noninteractive"

# Install dependencies
RUN apt-get update -y && \
	apt-get install bzip2 -y && \
	apt-get install wget -y && \
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
	chmod 755 Miniconda3-latest-Linux-x86_64.sh && \
	bash Miniconda3-latest-Linux-x86_64.sh -b -p /anaconda3 && \
	rm Miniconda3-latest-Linux-x86_64.sh && \
  wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh && \
  chmod 755 Miniconda2-latest-Linux-x86_64.sh && \
  bash Miniconda2-latest-Linux-x86_64.sh -b -p /anaconda && \
  rm Miniconda2-latest-Linux-x86_64.sh

# Change PATH for anaconda
ENV PATH=$PATH:/anaconda/bin/

RUN conda install -c conda-forge biopython -y && \
	conda install -c bioconda biom-format -y

RUN /anaconda3/bin/conda create -n py36 anaconda=2020.07 python=3.6 -y

RUN ln -s /anaconda3/envs/py36/bin/pip3 /anaconda/bin/pip3 && \
	ln -s /anaconda3/envs/py36/bin/python3 /anaconda/bin/python3

RUN /anaconda3/bin/conda update -n base -c defaults conda

RUN pip3 install scikit-learn==0.20.0 && \
	pip3 install HTSeq==0.9.1 && \
	pip3 install scipy==1.0.0 && \
	pip3 install numpy==1.16.4 && \
 pip install numpy && \
 pip install matplotlib

