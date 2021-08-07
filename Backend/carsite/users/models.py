from django.db import models
from django.core.validators import RegexValidator
from django.contrib.auth.models import User, AbstractBaseUser

class createprofile(AbstractBaseUser):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    carcolor = models.CharField(max_length=100)

    def __str__(self):
        return self.user.username


class profile(models.Model):

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    image = models.ImageField(default='default.jpg', upload_to='profile_pics')
    color_choices = ( ('Black', 'Black'), ('Silver', 'Silver'), ('Red', 'Red'),)
    carcolor = models.CharField(max_length=100, default='', choices=color_choices)
    payment_choices = ( ('Visa', 'Visa'), ('Mastercard', 'Mastercard'), ('Paybal', 'Paybal'),)
    paymentinfo = models.CharField(max_length=100, default='', choices=payment_choices)
    #phoneNumberRegex = RegexValidator(regex = r"^\+?1?\d{8,15}$")
    #phone = models.CharField(validators = [phoneNumberRegex], max_length = 10, unique = True)
    carmodels_choices = ( ('Toyota Chr', 'Toyota Chr'), ('Nissan Sunny', 'Nissan Sunny'), ('Toyota Corolla', 'Toyota Corolla'),)
    carmodels = models.CharField(max_length = 100, default='', choices=carmodels_choices)

    ins_choices = ( ('Egypt Inssurance', 'Egypt Inssurance'), 
                    ('Royal Egypt Inssurance', 'Royal Egypt Inssurance'), 
                    ('Takeful Egypt Inssurance', 'Takeful Egypt Inssurance'), )
    inssurance = models.CharField(max_length = 100, default='', choices=ins_choices)


    def __str__(self):
        return f'{self.user.username} profile'

class Customer(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, default=None , null=True)
    carcolor = models.CharField(max_length=100)
    paymentinfo = models.CharField(max_length=100)
    
    def __str__(self):
        return self.carcolor


class damageinfo(models.Model):
    carmodels = models.CharField(max_length = 100, default='')
    damageposition = models.CharField(max_length = 100, default='')
    broken_headlight = models.IntegerField()
    broken_taillight = models.IntegerField()
    dents = models.IntegerField()
    scratches = models.IntegerField()
    broken_glass = models.IntegerField()

    def __str__(self):
        return f'{self.carmodels} Damage Info'

class report(models.Model):
    username = models.ForeignKey(User, on_delete=models.CASCADE)
    isdamaged = models.CharField(max_length = 100, default='')
    carmodels =  models.CharField(max_length = 100, default='')
    damageposition = models.CharField(max_length = 100, default='')
    viewside = models.CharField(max_length = 100, default='')
    damageprice = models.IntegerField()
    image = models.ImageField(default='default.jpg', upload_to='profile_pics')

    def __str__(self):
        return f'{self.username} Report'
