# Generated by Django 3.1.2 on 2020-10-12 02:43

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0004_auto_20201012_1135'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='otherinfo',
            name='payment',
        ),
        migrations.RemoveField(
            model_name='payment',
            name='customer',
        ),
    ]
