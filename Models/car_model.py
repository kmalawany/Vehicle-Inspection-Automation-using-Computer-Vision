# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 12:59:42 2021

@author: Karim
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow_addons as tfa
import keras

train_dir = "D:\Graduation_project\\first model dataset\\train"
val_dir = "D:\Graduation_project\\first model dataset\\validation"


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                                                rotation_range=40,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                shear_range=0.2, 
                                                                zoom_range=0.2,
                                                                horizontal_flip=True,
                                                                fill_mode='nearest',
                                                                rescale=1./255
                                                                )

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                                                validation_split=0.33,
                                                                rescale=1./255
                                                                )



train_set = train_datagen.flow_from_directory(train_dir, class_mode = 'binary',
                                              batch_size = 32, 
                                              target_size=(150,150))

val_set = val_datagen.flow_from_directory(val_dir, class_mode = 'binary',
                                              batch_size = 32, 
                                              target_size=(150,150), subset = 'training')

test_set = val_datagen.flow_from_directory(val_dir, class_mode = 'binary',
                                              batch_size = 32, 
                                              target_size=(150,150), subset = 'validation')


    
def new_model():
    
    model = tf.keras.applications.DenseNet121(input_shape=(150, 150, 3), include_top=False)
    for layer in model.layers:
        layer.trainable = False
    
    last_layer = model.get_layer('relu')  
    x = last_layer.output

    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)

    x = layers.Dense(4096, activation='relu')(x)
    
    output = layers.Dense(1, 'sigmoid')(x)

    new_model = Model(model.input, output)
    
    return new_model
    
    
    
model = new_model()

lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5,verbose=1, mode='max')
    
METRICS = [
     'accuracy',
      tf.metrics.TruePositives(name='tp'),
      tf.metrics.FalsePositives(name='fp'),
      tf.metrics.TrueNegatives(name='tn'),
      tf.metrics.FalseNegatives(name='fn'), 
      tf.metrics.Precision(name='precision'),
      tf.metrics.Recall(name='recall'),
      #tfa.metrics.F1Score(name='F1_Score', num_classes=2),

      
]

adam = tf.keras.optimizers.Adam()

model.compile(optimizer = adam , loss ='binary_crossentropy', 
              metrics = METRICS)

history = model.fit(train_set, validation_data = val_set, epochs =15)

model.evaluate(test_set)
tf.keras.models.save_model(model, 'D:\Graduation_project\saved models\car_model.h5')
def preprocess_image(path):

    image = tf.keras.preprocessing.image.load_img(path=path, target_size=(150,150))
    array = keras.preprocessing.image.img_to_array(image)
    array = array / 255
    input_array = array[np.newaxis,:,:,:]
    return input_array



    
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(len(acc)) # Get number of epochs
rec = history.history['recall']
per = history.history['precision']
val_rec = history.history['val_recall']
val_perc = history.history['val_precision']

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------

plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'val'])
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------

plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend(['train loss', 'val loss'])
plt.title('Training and validation loss')
plt.figure()

plt.plot(epochs, rec, 'red')
plt.plot(epochs, val_rec, 'blue')
plt.xlabel('Epochs')
plt.ylabel('recall')
plt.legend(['train recall', 'val recall'])
plt.title('recall')
plt.figure()

plt.plot(epochs, per, 'r')
plt.plot(epochs, val_perc, 'b')
plt.xlabel('Epochs')
plt.ylabel('precision')
plt.legend(['train precision', 'val precision'])
plt.title('precision')
plt.figure()
    


