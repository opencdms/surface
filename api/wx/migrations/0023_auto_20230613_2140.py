# Generated by Django 3.2.13 on 2023-06-13 21:40

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('wx', '0022_maintenancereport_maintenancereportstationcomponent_stationcomponent_stationprofilecomponent_technic'),
    ]

    operations = [
        migrations.AlterField(
            model_name='station',
            name='begin_date',
            field=models.DateTimeField(null=True),
        ),
        migrations.AlterField(
            model_name='station',
            name='communication_type',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.DO_NOTHING, to='wx.stationcommunication'),
        ),
        migrations.AlterField(
            model_name='station',
            name='country',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.DO_NOTHING, to='wx.country'),
        ),
        migrations.AlterField(
            model_name='station',
            name='elevation',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='station',
            name='region',
            field=models.CharField(max_length=256, null=True),
        ),
        migrations.AlterField(
            model_name='station',
            name='watershed',
            field=models.CharField(max_length=256, null=True),
        ),
    ]