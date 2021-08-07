from django.urls import path
from . import views

urlpatterns = [
    path('home', views.home, name='main-home'),
    path('about/', views.about, name='main-about'),
    path('reports/', views.reports, name='main-reports'),
]