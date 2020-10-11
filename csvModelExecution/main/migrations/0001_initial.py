# Generated by Django 3.1.2 on 2020-10-10 09:08

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Customer',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('contract_type', models.IntegerField()),
                ('sex', models.CharField(max_length=30)),
                ('age', models.IntegerField()),
                ('pref', models.CharField(max_length=30)),
                ('payway', models.IntegerField()),
                ('identified_doc', models.IntegerField()),
                ('career', models.CharField(max_length=30)),
                ('domain', models.IntegerField()),
                ('region', models.CharField(max_length=30)),
                ('entry_age', models.IntegerField()),
                ('selling_method', models.CharField(max_length=30)),
                ('campaign', models.CharField(max_length=30)),
            ],
        ),
        migrations.CreateModel(
            name='OtherInfo',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('customer_rank', models.IntegerField()),
                ('arrears_at', models.IntegerField()),
                ('arrears_count', models.IntegerField()),
                ('bill_updage_count', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='Payment',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('amount', models.IntegerField()),
                ('paid_flg', models.IntegerField()),
            ],
        ),
    ]
