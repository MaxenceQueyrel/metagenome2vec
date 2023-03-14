# Distribution name
FROM ubuntu:18.04

LABEL org.opencontainers.image.authors="maxencequeyrel@hotmail.com"

ENV DEBIAN_FRONTEND="noninteractive"

ARG miniconda3_version=Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
ARG miniconda2_version=Miniconda2-py27_4.8.3-Linux-x86_64.sh
# Install dependencies
RUN apt-get update -y && \
	apt-get install git -y && \
	apt-get install bzip2 -y && \
	apt-get install wget -y && \
	wget https://repo.anaconda.com/miniconda/$miniconda3_version && \
	chmod 755 $miniconda3_version && \
	bash $miniconda3_version -b -p /anaconda3 && \
	rm $miniconda3_version && \
  wget https://repo.anaconda.com/miniconda/$miniconda2_version && \
  chmod 755 $miniconda2_version && \
  bash $miniconda2_version -b -p /anaconda && \
  rm $miniconda2_version

# Change PATH for anaconda
ENV PATH=$PATH:/anaconda/bin/

RUN conda install -c conda-forge biopython -y && \
	conda install -c bioconda biom-format -y

RUN /anaconda3/bin/conda create -n py36 anaconda=2020.07 python=3.6 -y

RUN ln -s /anaconda3/envs/py36/bin/pip3 /anaconda/bin/pip3 && \
	ln -s /anaconda3/envs/py36/bin/python3 /anaconda/bin/python3

RUN /anaconda3/bin/conda update -n base -c defaults conda

RUN pip3 install scikit-learn==0.20.0 && \
	# pip3 install HTSeq==0.9.1 && \
	pip3 install scipy==1.0.0 && \
	pip3 install numpy==1.16.4 && \
	pip3 install biopython==1.79 && \
	pip install numpy==1.16.0 && \
	pip install matplotlib==2.2.5

RUN git clone https://github.com/CAMI-challenge/CAMISIM.git && \
	git clone https://github.com/bcgsc/NanoSim.git

RUN mv /CAMISIM /opt && \
	mv /NanoSim /opt

# Add environment variables
ENV CAMISIM=/opt/CAMISIM \
	NANOSIM=/opt/NanoSim \
	TMP=/tmp


