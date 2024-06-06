FROM python:3.9-bullseye

WORKDIR /flask

ADD . /flask

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "/flask/scripts/data_export.py"]