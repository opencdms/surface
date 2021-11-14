FROM python:3.9.8-slim-bullseye
LABEL authors="Roberto Santos, Tadashi Igarashi"

ENV PYTHONUNBUFFERED 1
ENV DJANGO_SETTINGS_MODULE tempestas_api.settings

RUN apt-get update --fix-missing
RUN apt-get install -y g++ libgdal-dev libpq-dev libgeos-dev libproj-dev openjdk-17-jre vim wait-for-it

RUN adduser --disabled-password --gecos '' surface && mkdir /surface && chown -R surface /surface
WORKDIR /surface

RUN mkdir -p /data/documents
RUN mkdir -p /data/media
RUN mkdir -p /data/shared
RUN chown -R surface /data

COPY load_initial_data.sh .
RUN chmod +x load_initial_data.sh

USER surface

RUN pip install -U pip setuptools wheel
RUN pip install numpy==1.21.2

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt --no-warn-script-location
CMD ./startup.sh
