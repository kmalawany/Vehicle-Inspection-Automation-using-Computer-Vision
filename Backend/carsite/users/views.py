from django.contrib.auth.models import User
from django.db.models.base import Model
from django.shortcuts import render,redirect
from django.contrib.auth.forms import UserCreationForm 
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import UserRegisterForm, UserUpdateForm, ProfileUpdateForm, ExamForm
from .models import report, profile, damageinfo


def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}!')
            return redirect('createprofile')
    else:
        form = UserRegisterForm()
    return render(request, 'users/register.html', {'form': form})


@login_required
def profilee(request):
    if request.method == 'POST':
        u_form = UserUpdateForm(request.POST, instance=request.user)
        p_form = ProfileUpdateForm(request.POST, request.FILES, instance=request.user.profile)

        if u_form.is_valid() and p_form.is_valid():
            u_form.save()
            p_form.save()
            messages.success(request, f'Your account has been updated!')
            return redirect('profile')
    else:
        u_form = UserUpdateForm(instance=request.user)
        p_form = ProfileUpdateForm(instance=request.user.profile)

    context = {
        'u_form': u_form,
        'p_form': p_form
    }

    return render (request, 'users/profile.html', context)


@login_required
def profile_2(request):
    if request.method == 'POST':
        u_form = UserUpdateForm(request.POST, instance=request.user)
        p_form = ProfileUpdateForm(request.POST, request.FILES, instance=request.user.profile)

        if u_form.is_valid() and p_form.is_valid():
            u_form.save()
            p_form.save()
            messages.success(request, f'Your account has been updated!')
            return redirect('profile')
    else:
        u_form = UserUpdateForm(instance=request.user)
        p_form = ProfileUpdateForm(instance=request.user.profile)

    context = {
        'u_form' : u_form,
        'p_form' : p_form
    }

    return render (request, 'users/createprofile.html', context)


@login_required
def exam(request):
    if request.method == 'POST':
        form = ExamForm(request.POST)
        if form.is_valid():
            usname = request.user
            usname = usname.id
            instance=request.user.profile.carmodels

            postiosn = form.cleaned_data.get('position')
            view = form.cleaned_data.get('viewside')
            
            dg = damageinfo.objects.filter(carmodels=instance)
            dgg = dg.values(postiosn)

            report.objects.create(username=User.objects.get(id=usname), 
                                  isdamaged=form.cleaned_data.get('isdamaged'),
                                  carmodels=instance, 
                                  damageposition=postiosn,
                                  viewside=view,
                                  damageprice=dgg,
                                  image=form.cleaned_data.get('image'))

            messages.success(request, f'Report created for {dgg}!')
            return redirect('exam')
    else:
        form = ExamForm()
    return render(request, 'users/exam.html', {'form': form})


@login_required
def showreports(request):
    context = {
        'Reports': report.objects.all()
    }
    return render(request, 'users/showreports.html', context)







