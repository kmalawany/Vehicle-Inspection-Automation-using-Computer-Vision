# Generated by Django 3.1.7 on 2021-07-10 17:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('carmodels', '0002_auto_20210710_1924'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imagedb',
            name='image',
            field=models.ImageField(default='default.jpg', upload_to='profile_pics'),
        ),
    ]
