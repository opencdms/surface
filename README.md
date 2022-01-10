# Surface

## Run Docker

docker-compose up

## Run and Stop the project

docker-compose -f docker-compose-dev.yml up

docker-compose -f docker-compose-dev.yml stop

## Production env

in api file add the production env, you can find a example in "./api/production.env.example", add the values of variables, in hosts and ports you can put 0

## Generate Docker Images
docker-compose -f docker-compose-dev.yml build

## Running with Docker

docker-compose -f docker-compose-dev.yml up postgres cache redis api

## Load initial data

docker-compose -f docker-compose-dev.yml exec api bash load_initial_data.sh

### if you're using windows

docker-compose -f docker-compose-dev.yml exec api bash

python manage.py migrate

python manage.py loaddata /surface/fixtures/*

## Create superuser

docker-compose -f docker-compose-dev.yml exec api python manage.py createsuperuser

## Collect static files before build for production

docker-compose -f docker-compose-prd.yml -p surface_new exec api bash load_initial_data.sh

docker-compose -f docker-compose-prd.yml -p surface_new exec api python manage.py collectstatic --noinput

## Loading data

docker-compose -f docker-compose-dev.yml exec postgres pg_restore -U dba -d surface_db /data/shared/dump_surface_20211114.dump

docker-compose -f docker-compose-dev.yml exec postgres psql -U dba -d surface_db -c "\COPY raw_data FROM '/data/shared/dump_raw_data_20211130.csv' WITH DELIMITER ',' CSV HEADER;"

## Access DB manually

docker-compose -f docker-compose-dev.yml exec postgres psql -U dba -d surface_db