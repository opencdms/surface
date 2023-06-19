# Generated by Django 3.2.13 on 2023-06-19 19:18

import ckeditor.fields
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('wx', '0023_auto_20230613_2140'),
    ]

    operations = [
        migrations.CreateModel(
            name='EquipmentType',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('name', models.CharField(max_length=64)),
                ('description', models.CharField(max_length=256)),
                ('report_template', ckeditor.fields.RichTextField(blank=True, null=True)),
            ],
            options={
                'verbose_name': 'equipment type',
                'verbose_name_plural': 'equipment types',
            },
        ),
        migrations.CreateModel(
            name='Manufacturer',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('name', models.CharField(max_length=64)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Equipment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('model', models.CharField(max_length=64)),
                ('serial_number', models.CharField(max_length=64)),
                ('acquired', models.DateField()),
                ('first_deployed', models.DateField(blank=True, null=True)),
                ('equipment_type', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='wx.equipmenttype')),
                ('manufacturer', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='wx.manufacturer')),
            ],
            options={
                'verbose_name': 'equipment',
                'verbose_name_plural': 'equipments',
                'unique_together': {('equipment_type', 'serial_number')},
            },
        ),
    ]
