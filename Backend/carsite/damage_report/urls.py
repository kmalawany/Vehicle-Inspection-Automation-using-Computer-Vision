from django.urls import path
from . import views


urlpatterns = [
    path('', views.model_form, name='image'),
    path('damage', views.model_form, name='home'),

]