# Generated by Django 3.1.7 on 2021-07-09 20:26

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
            name='damageinfo',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('carmodels', models.CharField(default='', max_length=100)),
                ('damageposition', models.CharField(default='', max_length=100)),
                ('brokenheadlight', models.IntegerField()),
                ('brokentaillight', models.IntegerField()),
                ('dents', models.IntegerField()),
                ('scratches', models.IntegerField()),
                ('brokenglass', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='report',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('isdamaged', models.CharField(default='', max_length=100)),
                ('carmodels', models.CharField(default='', max_length=100)),
                ('damageposition', models.CharField(default='', max_length=100)),
                ('viewside', models.CharField(default='', max_length=100)),
                ('damageprice', models.IntegerField()),
                ('image', models.ImageField(default='default.jpg', upload_to='profile_pics')),
                ('username', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='profile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(default='default.jpg', upload_to='profile_pics')),
                ('carcolor', models.CharField(choices=[('Black', 'Black'), ('Silver', 'Silver'), ('Red', 'Red')], default='', max_length=100)),
                ('paymentinfo', models.CharField(choices=[('Visa', 'Visa'), ('Mastercard', 'Mastercard'), ('Paybal', 'Paybal')], default='', max_length=100)),
                ('carmodels', models.CharField(choices=[('Toyota Chr', 'Toyota Chr'), ('Nissan Sunny', 'Nissan Sunny'), ('Toyota Corolla', 'Toyota Corolla')], default='', max_length=100)),
                ('inssurance', models.CharField(choices=[('Egypt Inssurance', 'Egypt Inssurance'), ('Royal Egypt Inssurance', 'Royal Egypt Inssurance'), ('Takeful Egypt Inssurance', 'Takeful Egypt Inssurance')], default='', max_length=100)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Customer',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('carcolor', models.CharField(max_length=100)),
                ('paymentinfo', models.CharField(max_length=100)),
                ('user', models.OneToOneField(default=None, null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='createprofile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('password', models.CharField(max_length=128, verbose_name='password')),
                ('last_login', models.DateTimeField(blank=True, null=True, verbose_name='last login')),
                ('carcolor', models.CharField(max_length=100)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'abstract': False,
            },
        ),
    ]