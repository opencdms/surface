# Generated by Django 3.2.13 on 2024-09-28 02:43

import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('wx', '0028_auto_20240814_1808'),
    ]

    operations = [
        migrations.CreateModel(
            name='CountryISOCode',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('name', models.CharField(max_length=256, unique=True)),
                ('description', models.CharField(blank=True, max_length=256, null=True)),
                ('notation', models.CharField(max_length=256, unique=True)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='UTCOffsetMinutes',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('hours', models.CharField(max_length=256, unique=True)),
                ('minutes', models.IntegerField()),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='WMOReportingStatus',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('name', models.CharField(max_length=256, unique=True)),
                ('description', models.CharField(blank=True, max_length=256, null=True)),
                ('notation', models.CharField(blank=True, max_length=256, null=True)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.AddField(
            model_name='station',
            name='international_station',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='station',
            name='wigos_part_1',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='station',
            name='wigos_part_3',
            field=models.IntegerField(blank=True, null=True, validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(65534)]),
        ),
        migrations.AddField(
            model_name='station',
            name='wigos_part_4',
            field=models.CharField(blank=True, max_length=16, null=True, unique=True),
        ),
        migrations.AlterField(
            model_name='station',
            name='is_automatic',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='station',
            name='reporting_status',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.DO_NOTHING, to='wx.wmoreportingstatus'),
        ),
        migrations.AddField(
            model_name='station',
            name='wigos_part_2',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.DO_NOTHING, to='wx.countryisocode'),
        ),
    ]
