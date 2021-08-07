from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from .models import DamageReport

def home(request):
    context = {
        'Reports': DamageReport.objects.all()
    }
    return render(request, 'main/home.html', context)

def about(request):
    return render(request, 'main/about.html', {'title': 'About'})

@login_required
def reports(request):
    context = {
        'Reports': DamageReport.objects.all()
    }
    return render(request, 'main/reports.html', context)
