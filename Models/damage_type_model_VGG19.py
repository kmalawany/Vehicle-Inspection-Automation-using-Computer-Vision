# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 12:41:33 2021

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
from keras import backend as K
import keras
import pandas as pd
import seaborn as sns
tf.__version__

train_dir = "D:\\Graduation_project\\damage type"
test_dir = 'D:\Graduation_project\Damage type not collected'


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                                          rotation_range=40,
                                                          width_shift_range=0.2,
                                                          height_shift_range=0.2,
                                                          shear_range=0.2, 
                                                          zoom_range=0.2,
                                                          horizontal_flip=True,
                                                          fill_mode='nearest',
                                                          rescale=1./255,
                                                          validation_split=0.3)

# val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.3)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


train_set = train_datagen.flow_from_directory(train_dir,
                                              class_mode = 'categorical',
                                              batch_size = 64, 
                                              target_size=(150,150),
                                              subset='training')

val_set = train_datagen.flow_from_directory(train_dir,
                                            class_mode = 'categorical',
                                            batch_size = 64, 
                                            target_size=(150,150),
                                            subset='validation')

test_set = test_datagen.flow_from_directory(test_dir,
                                            class_mode = 'categorical',
                                            batch_size = 64, 
                                            target_size=(150,150))


label_map = (train_set.class_indices)
labels = ['Broken glass', 'Broken headlights', 'Broken taillights', 'Dents', 'scratches']


#count size of each class
broken_glass = len(os.listdir(train_dir + '\\Broken Glass'))
broken_headlights = len(os.listdir(train_dir + '\\Broken Headlights'))
broken_taillight = len(os.listdir(train_dir + '\\Broken Tail Lights'))
dents = len(os.listdir(train_dir + '\\Dents'))
scratches = len(os.listdir(train_dir + '\\Scratches'))
                         

def bar_plot_classes():
    labels = ['Broken_glass', 'Broken_headlights', 'broken_taillight', 'dents', 'scratches']
    values = [broken_glass,broken_headlights,broken_taillight,dents,scratches]

    x_pos = [i for i, _ in enumerate(labels)]

    plt.bar(x_pos, values)
    plt.xlabel("Classes")
    plt.ylabel("images")
    plt.title("Damage types classes")
    plt.xticks(x_pos, labels)

    plt.show()
 
# bar_plot_classes()
 
def compute_class_freqs(labels):
       
    N = labels.shape[0]
    
    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies

    return positive_frequencies, negative_frequencies


notlabels = train_set.labels
one_hot_labels = keras.utils.to_categorical(notlabels, num_classes=5)

np.shape(one_hot_labels)
pos_freq, neg_freq =compute_class_freqs(one_hot_labels)

pos_weights = neg_freq
neg_weights = pos_freq

# plt.bar(x=labels, height=np.mean(one_hot_labels, axis=0))
# plt.title("Frequency of Each Class")
# plt.show()


def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    
    def weighted_loss(y_true, y_pred):
      
        # initialize loss to zero
        loss = 0.0
        

        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class
            loss_pos = -1 * K.mean(pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon))
            loss_neg = -1 * K.mean(neg_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon))
            loss += loss_pos + loss_neg
        
        return loss
    
    return weighted_loss



def dense_block(units, rate, Lambda, inputs):
    
    x = layers.Dense(units, kernel_regularizer=tf.keras.regularizers.l2(Lambda))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(rate)(x)
    
    return x



def create_model():
    model = tf.keras.applications.VGG19(input_shape=(150, 150, 3), include_top=False)
    for layer in model.layers:
        layer.trainable = False
    
    last_layer = model.get_layer('block5_pool')
  
    x = last_layer.output

    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)


    x = dense_block(2048, 0.7, 0.0001, x)
    x = dense_block(1024, 0.6, 0.0001, x)
    x = dense_block(512, 0.5, 0.0001, x)
    x = dense_block(256, 0.4, 0.0001, x)


    output = layers.Dense(5, 'sigmoid')(x)

    new_model = Model(model.input, output)
    
    return new_model

model = create_model()

METRICS = [ 
     'accuracy',
      tf.metrics.TruePositives(name='tp'),
      tf.metrics.FalsePositives(name='fp'),
      tf.metrics.TrueNegatives(name='tn'),
      tf.metrics.FalseNegatives(name='fn'), 
      tf.metrics.Precision(name='precision'),
      tf.metrics.Recall(name='recall'),
      tfa.metrics.F1Score(name='F1_Score', num_classes=5),

      
]

adam = tf.keras.optimizers.Adam()

lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=5, verbose=2, mode='max')

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)


model.compile(optimizer = adam , loss = get_weighted_loss(pos_weights, neg_weights), 
                  metrics = METRICS)

model.summary()
history = model.fit(train_set, validation_data = val_set, epochs =50, callbacks=lr_reduce)


# model.save_weights('D:\Graduation_project\Damage type model\Damage type model')
# model.load_weights('D:\Graduation_project\Damage type model\Damage type model')
# model.summary()


model.evaluate(test_set)


def preprocess_image(path):

    image = tf.keras.preprocessing.image.load_img(path=path, target_size=(150,150))
    array = keras.preprocessing.image.img_to_array(image)
    array = array / 255
    input_array = array[np.newaxis,:,:,:]
    return input_array


image = preprocess_image('D:\Graduation_project\\Damaged-car-sml.jpg')

predict = model.predict(image)




def decode_labels(predict, labels):
    
    predict[predict >= 0.5] = 1
    predict[predict < 0.5] = 0
    
    lab = []
    
    List = np.reshape(predict,(5, 1))
    flatten = List.flatten()
    pred_list = flatten.tolist()
    
    for i in range(len(pred_list)):
        if pred_list[i] == 1.0:          
            lab.append(labels[i])
        
    return lab

decode = decode_labels(predict, labels)

print(decode)  
    


    
    
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


