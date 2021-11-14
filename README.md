# Surface

## Running with Docker

docker-compose -f docker-compose-dev.yml up postgres cache redis api

## Load initial data

docker-compose -f docker-compose-dev.yml exec api bash load_initial_data.sh

## Create superuser

docker-compose -f docker-compose-dev.yml exec api python manage.py createsuperuser

## Collect static files before build for production

docker-compose -f docker-compose-prd.yml exec api python manage.py collectstatic --noinput
