from django.urls import path
from . import views

urlpatterns = [
    path('', views.imageml, name='image'),
    path('', views.imageml, name='home'),

]