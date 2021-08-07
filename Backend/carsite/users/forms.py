from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.db import models
from django.db.models import fields
from django.forms.models import ModelForm
from django.shortcuts import redirect
from .models import profile, createprofile, Customer, report


class CustomerForm(ModelForm):
    class Meta:
        model = Customer
        fields = ['carcolor', 'paymentinfo']


class CreateProfile(forms.ModelForm):

    class Meta:
        model = createprofile
        fields = ['carcolor']


class UserProfile(forms.ModelForm):
    class Meta:
        model = profile
        fields = ['image']


class UserRegisterForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']


class UserUpdateForm(forms.ModelForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'email']


class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = profile
        fields = ['carcolor', 'paymentinfo', 'inssurance', 'carmodels', 'image']


class ReportForm(forms.ModelForm):
    class Meta:
        model = report
        fields = ['username', 'carmodels', 'damageposition']


class ExamForm(forms.ModelForm):
    viewside = forms.CharField(max_length=100)
    position = forms.CharField(max_length=100)
    isdamaged = forms.CharField(max_length=100)
    image = models.ImageField(default='default.jpg', upload_to='profile_pics')

    class Meta:
        model = report
        fields = ['viewside', 'position', 'image', 'isdamaged']
