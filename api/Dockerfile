FROM python:3.9.7-slim-bullseye
LABEL authors="Roberto Santos, Tadashi Igarashi"

ENV PYTHONUNBUFFERED 1
ENV DJANGO_SETTINGS_MODULE tempestas_api.settings

RUN apt-get update --fix-missing
RUN apt-get install -y g++ libgdal-dev libpq-dev libgeos-dev libproj-dev openjdk-17-jre vim wait-for-it

RUN adduser --disabled-password --gecos '' surface && mkdir /surface && chown -R surface /surface
WORKDIR /surface

USER surface
RUN pip install -U pip
RUN pip install numpy==1.21.2 --no-warn-script-location

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt --no-warn-script-location

CMD python manage.py migrate && python manage.py runserver 0.0.0.0:8000
