# Generated by Django 3.1.2 on 2020-10-12 09:43

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0002_auto_20201012_1804'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='otherinfo',
            name='payment',
        ),
        migrations.AddField(
            model_name='otherinfo',
            name='customer',
            field=models.OneToOneField(null=True, on_delete=django.db.models.deletion.SET_NULL, to='main.customer'),
        ),
    ]
