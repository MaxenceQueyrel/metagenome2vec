#Distribution name
FROM maxence27/deep_learning:cuda114-cudnn8

LABEL org.opencontainers.image.authors="maxencequeyrel@hotmail.com"

ARG DEBIAN_FRONTEND="noninteractive"

RUN apt update -y && \
  apt install default-jdk -y && \
  apt install git -y

RUN pip install pyspark==3.3.2 && \
  pip install dill==0.3.6 && \
  pip install Bio==1.5.3 && \
  pip install pyarrow==6.0.1 && \
  pip install fasttext==0.9.2 && \
  pip install Cython==0.29.32 && \
  pip install matplotlib==3.6.2 && \
  pip install seaborn==0.10.0 && \
  pip install ete3==3.1.2 && \
  pip install h2o==3.38.0.4 && \
  pip install logger==1.4 && \
  pip install pandas==1.5.0 && \
  pip install numpy==1.24.3 && \
  pip install scikit_bio==0.5.8 && \
  pip install scikit_learn==1.2.1 && \
  pip install scipy==1.9.0 && \
  pip install seaborn==0.12.2 && \
  pip install tqdm==4.62.3 && \
  pip install umap-learn==0.5.3


RUN git clone https://github.com/rmenegaux/fastDNA.git && \
	mv fastDNA /opt && \
	cd /opt/fastDNA && \
	make

# Add metagenome2vec package
COPY . /opt/metagenome2vec
WORKDIR /opt/metagenome2vec
 

# Add environment variables
ENV CAMISIM_PATH=/opt/CAMISIM \
	NANOSIM_PATH=/opt/NanoSim \
	METAGENOME2VEC_PATH=/opt/metagenome2vec \
	TMP=/tmp \
	FASTDNA=/opt/fastDNA

# 
# Pipeline==0.1
#   pip install biopython==1.79 && \


