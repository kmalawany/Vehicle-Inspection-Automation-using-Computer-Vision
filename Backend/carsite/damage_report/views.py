from django.contrib import messages
from django.db.models import Max
from django.shortcuts import render
import tensorflow as tf
import keras
import numpy as np
from carmodels.models import imagedb
from users.models import report
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User

# Create your views here.


# preprocess input image
def preprocess_image(path):
    image = tf.keras.preprocessing.image.load_img(path=path, target_size=(150, 150))
    array = keras.preprocessing.image.img_to_array(image)
    array = tf.cast(array, tf.float32)
    array = array / 255
    input_array = array[np.newaxis, :, :, :]

    return input_array


def decode_labels_damage_type(predict):
    labels = ['broken_glass', 'broken_headlight', 'broken_taillight', 'dents', 'scratches']

    predict[predict >= 0.5] = 1
    predict[predict < 0.5] = 0

    lab = []

    List = np.reshape(predict, (5, 1))
    flatten = List.flatten()
    pred_list = flatten.tolist()

    for i in range(len(pred_list)):
        if pred_list[i] == 1.0:
            lab.append(labels[i])
    output = ', '.join(str(i) for i in lab)

    return output


def decode_labels_damaged(predict):
    predict[predict >= 0.5] = 1
    predict[predict < 0.5] = 0

    List = np.reshape(predict, (1, 1))
    flatten = List.flatten()
    pred_list = flatten.tolist()

    if pred_list[0] == 1.0:
        label = 'Not damaged'
    else:
        label = 'damaged'

    return label


def decode_labels_car_view(predict):
    List = np.reshape(predict, (3, 1))
    flatten = List.flatten()
    pred_list = flatten.tolist()

    dic = {'Back view': pred_list[0], 'Front view': pred_list[1], 'side view': pred_list[2]}
    max_value = max(dic.values())
    for keys, values in dic.items():
        if values == max_value:
            newk = keys
            break

    return newk


def decode_labels_car_model(predict):
    predict[predict >= 0.5] = 1
    predict[predict < 0.5] = 0

    List = np.reshape(predict, (1, 1))
    flatten = List.flatten()
    pred_list = flatten.tolist()

    if pred_list[0] == 1.0:
        label = 'not car'
    else:
        label = 'car'

    return label


def damage_type_model():
    model = tf.keras.models.load_model("D:\\Graduation_project\\saved models\\damage_type_model.h5", compile=False)
    return model


def damaged_model():
    model = tf.keras.models.load_model("D:\\Graduation_project\\saved models\\damaged_model.h5", compile=False)
    return model


def car_view():
    model = tf.keras.models.load_model("D:\\Graduation_project\\saved models\\car_view_model.h5", compile=False)
    return model


def car_model():
    model = tf.keras.models.load_model("D:\\Graduation_project\\saved models\\car_model.h5", compile=False)
    return model


@login_required
def model_form(request):
    image_obj = imagedb()
    path = str(image_obj.get_image_name())
    abs_path = 'D:/Graduation_project/django_project/carsite/media/' + path
    show_image = 'media' + '/' + path
    # load models
    model1 = car_model()
    model2 = car_view()
    model3 = damaged_model()
    model4 = damage_type_model()

    # preprocess image
    image = preprocess_image(abs_path)

    # get models predications
    model1_arr = model1.predict(image)
    model2_arr = model2.predict(image)
    model3_arr = model3.predict(image)
    model4_arr = model4.predict(image)

    # decode outputs to labels
    model1_output = decode_labels_car_model(model1_arr)
    model2_output = decode_labels_car_view(model2_arr)
    model3_output = decode_labels_damaged(model3_arr)
    model4_output = decode_labels_damage_type(model4_arr)

    # check if image is a car
    if model1_output == 'car':
        pass
    else:
        model2_output = 'Not car'
        model3_output = 'Not car'
        model4_output = 'Not car'

   # check if image is damaged
    if model3_output == 'damaged':
        pass
    else:

        model4_output = 'Not damaged'

    instance = request.user.profile.carmodels

    usname = request.user
    usname = usname.id

    if model1_output == 'car' and model3_output == 'damaged':

        report.objects.create(username=User.objects.get(id=usname),
                              isdamaged=model3_output,
                              carmodels=instance,
                              damageposition=model4_output,
                              viewside=model2_output,
                              damageprice=0,
                              image=abs_path)
    else:
        report.objects.create(username=User.objects.get(id=usname),
                              isdamaged='Not Damaged',
                              carmodels=instance,
                              damageposition='None',
                              viewside='None',
                              damageprice=0,
                              image=abs_path)
    #img = imagedb.objects.all().latest(image)
    report_output = {'model1': model1_output,
                     'model2': model2_output,
                     'model3': model3_output,
                     'model4': model4_output,
                     'imgpath': show_image}
    return render(request, 'damage_report/damage_report.html', context=report_output)
