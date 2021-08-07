# Generated by Django 3.1.6 on 2021-07-05 07:18

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='DamageInfo',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('carmodel', models.CharField(max_length=200)),
                ('brokenglass', models.IntegerField()),
                ('dents', models.IntegerField()),
                ('scratches', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='DamageReport',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('damagestatus', models.BooleanField()),
                ('DamageInfo', models.CharField(max_length=200)),
                ('driver', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]