from django.shortcuts import render, redirect
from .models import imagedb
from django.contrib import messages
from .forms import ImageForm
from django.contrib.auth.decorators import login_required


@login_required
def imageml(request):
    if request.method == 'POST':
        i_form = ImageForm(request.POST, request.FILES,)
        if i_form.is_valid():
            image = i_form.cleaned_data.get('image')
            imagedb.objects.create(image=i_form.cleaned_data.get('image'))
            return redirect('/damage')

    else:
        i_form = ImageForm()

    context = {
        'i_form': i_form,
    }
    return render(request, 'carmodels/imageml.html', context)


@login_required
def get_image_from_user(request):
    if request.method == 'POST':
        i_form = ImageForm(request.POST, request.FILES, )
        if i_form.is_valid():
            image = i_form.cleaned_data.get('image')
            imagedb.objects.create(image=i_form.cleaned_data.get('image'))
            messages.success(request, f'image created for {image}!')

    return image



def image_name(request):
    image = imagedb.get_image_name()
    dicti = {'name': image}
    return render(request, 'carmodels/models_form.html', context=dicti)


