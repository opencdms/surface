# Generated by Django 3.2.13 on 2022-11-04 00:33

from django.db import migrations, models
import django.db.models.deletion
import timescale.db.models.fields


class Migration(migrations.Migration):

    dependencies = [
        ('wx', '0017_auto_20220803_1529'),
    ]

    operations = [
        migrations.AddField(
            model_name='noaadcp',
            name='config_data',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='noaadcp',
            name='config_file',
            field=models.FileField(blank=True, null=True, upload_to=''),
        ),
        migrations.AddField(
            model_name='stationfileingestion',
            name='is_highfrequency_data',
            field=models.BooleanField(default=True),
        ),
        migrations.CreateModel(
            name='HightFrequencyData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('datetime', timescale.db.models.fields.TimescaleDateTimeField(interval='1 day')),
                ('measured', models.FloatField()),
                ('station', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='wx.station')),
                ('variable', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='wx.variable')),
            ],
        ),
        migrations.CreateModel(
            name='ElementDecoder',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('element_name', models.CharField(max_length=64)),
                ('decoder', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.DO_NOTHING, to='wx.decoder')),
                ('variable', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.DO_NOTHING, to='wx.variable')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.AddConstraint(
            model_name='hightfrequencydata',
            constraint=models.UniqueConstraint(fields=('datetime', 'station', 'variable'), name='unique_datetime_station_id_variable_id'),
        ),
    ]
