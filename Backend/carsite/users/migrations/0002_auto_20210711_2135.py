# Generated by Django 3.1.7 on 2021-07-11 19:35

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='damageinfo',
            old_name='brokenglass',
            new_name='broken_glass',
        ),
        migrations.RenameField(
            model_name='damageinfo',
            old_name='brokenheadlight',
            new_name='broken_headlight',
        ),
        migrations.RenameField(
            model_name='damageinfo',
            old_name='brokentaillight',
            new_name='broken_taillight',
        ),
    ]