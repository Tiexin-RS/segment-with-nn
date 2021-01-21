FROM tensorflow/tensorflow:2.4.0-gpu

WORKDIR /opt/segelectri
COPY ./requirements.txt requirements.txt
RUN apt update && apt install -y git
RUN python3 -m pip install -r requirements.txt -i https://pypi.doubanio.com/simple

ENV HOME=/opt/env
ENV SHELL=/bin/bash
RUN mkdir -p ${HOME} && chmod 777 ${HOME} && chmod 777 /opt/
