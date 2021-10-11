# Surface

## Running with Docker

docker-compose up postgres cache redis api

## Load initial data

docker-compose exec api python manage.py migrate 
docker-compose exec api bash load_initial_data.sh

## Create superuser

docker-compose exec api python manage.py createsuperuser