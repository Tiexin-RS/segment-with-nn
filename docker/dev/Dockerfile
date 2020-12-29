FROM tensorflow/tensorflow:latest-gpu

ENV HOME=/opt/env
ENV SHELL=/bin/bash

WORKDIR /opt/segelectri
COPY ./requirements.txt requirements.txt
RUN mkdir -p ${HOME} && chmod 777 ${HOME}
RUN apt update && apt install -y git
RUN python3 -m pip install -r requirements.txt
