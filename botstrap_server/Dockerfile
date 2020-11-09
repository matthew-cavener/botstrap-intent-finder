FROM tensorflow/tensorflow:2.3.1
# RUN apt-get update -q -y && apt-get install -q -y python3-dev

WORKDIR /usr/src/app/botstrap_server
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . ./

ENV TFHUB_CACHE_DIR=/usr/src/app/botstrap_server/hub_modules
ENV PYTHONUNBUFFERED=0

RUN useradd -ms /bin/bash botstrap
USER botstrap
EXPOSE 8080