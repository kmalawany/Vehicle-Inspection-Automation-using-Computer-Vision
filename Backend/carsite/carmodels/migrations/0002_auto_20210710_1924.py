# Generated by Django 3.1.7 on 2021-07-10 17:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('carmodels', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imagedb',
            name='image',
            field=models.ImageField(default='default.jpg', upload_to='media/profile_pics'),
        ),
    ]
