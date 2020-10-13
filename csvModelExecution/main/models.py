from django.db import models

class Customer(models.Model):
    id = models.AutoField(primary_key=True)
    contract_type = models.IntegerField()
    sex = models.CharField(max_length=30)
    age = models.IntegerField()
    pref = models.CharField(max_length=30)
    payway = models.IntegerField()
    identified_doc = models.IntegerField()
    career = models.CharField(max_length=30)
    domain = models.IntegerField()
    region = models.CharField(max_length=30)
    entry_age = models.IntegerField()
    selling_method = models.CharField(max_length=30)
    campaign = models.CharField(max_length=30)

class Payment(models.Model):
    id = models.AutoField(primary_key=True)
    amount = models.IntegerField()
    paid_flg = models.IntegerField()

class OtherInfo(models.Model):
    id = models.AutoField(primary_key=True)
    customer_rank = models.IntegerField()
    arrears_at = models.IntegerField()
    arrears_count = models.IntegerField()
    bill_updage_count = models.IntegerField()
