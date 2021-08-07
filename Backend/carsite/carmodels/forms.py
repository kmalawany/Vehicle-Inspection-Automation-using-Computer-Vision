from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.db import models
from django.db.models import fields
from django.forms.models import ModelForm
from django.shortcuts import redirect
from .models import imagedb


class ImageForm(forms.ModelForm):
    image = forms.ImageField()

    class Meta:
        model = imagedb
        fields = ['image']
