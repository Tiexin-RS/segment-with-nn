FROM tensorflow/tensorflow:2.4.0-gpu

WORKDIR /opt/segelectri
COPY ./requirements.txt requirements.txt
RUN apt update && apt install -y git build-essential nghttp2 libnghttp2-dev libssl-dev iputils-ping
RUN python3 -m pip install -r requirements.txt -i https://pypi.doubanio.com/simple

ARG ARG_UID=1001
ARG ARG_GID=1001

ENV HOME=/opt/env
ENV SHELL=/bin/bash
RUN chown -R ${ARG_UID}:${ARG_GID} /opt/

USER ${ARG_UID}:${ARG_GID}
RUN mkdir -p ${HOME} 
