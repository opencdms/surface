# Generated by Django 3.2.9 on 2022-06-23 10:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('wx', '0015_auto_20220609_1945'),
    ]

    operations = [
        migrations.AddField(
            model_name='variable',
            name='persistence_window',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='variable',
            name='persistence_window_hourly',
            field=models.IntegerField(blank=True, null=True),
        ),
    ]