#!/bin/sh

wait-for-it postgres:5432
exec /home/surface/.local/bin/gunicorn -b 0.0.0.0:8000 --workers=$NUMBER_OF_GUNICORN_WORKERS tempestas_api.wsgi:application