# Generated by Django 3.2.9 on 2022-04-11 17:40

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('wx', '0007_auto_20220411_1307'),
    ]

    operations = [
        migrations.AddField(
            model_name='station',
            name='country_link',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.DO_NOTHING, to='wx.country'),
        ),
    ]