FROM geopython/pygeoapi:latest

WORKDIR /pygeoapi-flask

ADD . /pygeoapi-flask

RUN pip install -r ./requirements.txt

EXPOSE 5000

CMD ["python", "/pygeoapi-flask/app.py"]