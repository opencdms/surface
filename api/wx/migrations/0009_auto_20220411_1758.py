# Generated by Django 3.2.9 on 2022-04-11 17:58

from django.db import migrations
from django.core.exceptions import ObjectDoesNotExist


def link_countries(apps, schema_editor):
    Station = apps.get_model('wx', 'Station')
    Country = apps.get_model('wx', 'Country')
    for station in Station.objects.all():
        try:
            country = Country.objects.get(name=station.country)
            station.country_link = country
            station.save()
        except ObjectDoesNotExist:
            pass

class Migration(migrations.Migration):

    dependencies = [
        ('wx', '0008_station_country_link'),
    ]

    operations = [
        migrations.RunPython(link_countries)
    ]
