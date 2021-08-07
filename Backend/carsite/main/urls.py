from django.urls import path
from . import views
from carmodels import views as sv
urlpatterns = [
    path('', sv.imageml, name='main-home'),
    path('about/', views.about, name='main-about'),
    path('reportsss/', views.reports, name='main-reports'),
]