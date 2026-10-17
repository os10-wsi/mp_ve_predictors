# DeepSequence was published against Python 2.7 and Theano, which cannot
# coexist with anything else in this repository, so it runs in a container.
#
#   docker build -t vepbench-deepsequence envs/deepsequence
#
# The entry point reads an alignment and a variant list and writes a CSV of
# ELBO differences, which is what scripts/runners/deepsequence.py expects.
FROM python:2.7-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential gfortran libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy==1.16.6 scipy==1.2.3 theano==1.0.5 pandas==0.24.2

RUN git clone --depth 1 https://github.com/debbiemarkslab/DeepSequence.git /opt/DeepSequence
WORKDIR /opt/DeepSequence

COPY score_deepsequence.py /opt/score_deepsequence.py
ENV THEANO_FLAGS=floatX=float32,device=cpu
ENTRYPOINT ["python", "/opt/score_deepsequence.py"]
