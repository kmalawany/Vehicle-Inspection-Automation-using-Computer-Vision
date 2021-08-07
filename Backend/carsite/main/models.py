from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

class DamageInfo(models.Model):
    carmodel = models.CharField(max_length=200)
    brokenglass = models.IntegerField()
    dents= models.IntegerField()
    scratches = models.IntegerField()

class DamageReport(models.Model):
    damagestatus = models.BooleanField()
    DamageInfo = models.CharField(max_length=200)
    dateposted = models.DateTimeField(default=timezone.now)
    driver = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.DamageInfo