import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import cv2
from PIL import Image as im
from glob import glob
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.models import Model
from keras.regularizers import l2
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping


path = os.getcwd()
train_path = path + "/train/"
test_path = path + "/test/"
classes = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']

train_data = []
train_labels = []
for i in classes:
    dir = train_path + i
    for j in os.listdir(dir):
        img_path = dir + '/' + j
        img = cv2.imread(img_path,-1)
        img = cv2.resize(img,(224,224),interpolation = cv2.INTER_NEAREST)
        train_data.append(img)
        train_labels.append(i)

test_data = []
test_labels = []
for i in classes:
    dir = test_path + i
    for j in os.listdir(dir):
        img_path = dir + '/' + j
        img = cv2.imread(img_path,-1)
        img = cv2.resize(img,(224,224),interpolation = cv2.INTER_NEAREST)
        test_data.append(img)
        test_labels.append(i)

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

training_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')
training_data = training_datagen.flow_from_directory(train_path,
                                      target_size=(224, 224),
                                      batch_size=32,
                                      class_mode='categorical')

testing_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')
testing_data = testing_datagen.flow_from_directory(test_path,
                                      target_size=(224, 224),
                                      batch_size=32,
                                      class_mode='categorical')

vgg = VGG16(input_shape = (224, 224, 3), weights='imagenet', include_top=False)
for layers in vgg.layers:
    layers.trainable = False

#Create new transfer learning model   
flatten_layer = Flatten()
dropout = Dropout(0.2)
dense_layer_1 = Dense(50, activation='relu')
dense_layer_2 = Dense(20, activation='relu')
prediction_layer = Dense(7, activation='softmax', kernel_regularizer = l2(0.2))

model = keras.Sequential([
    vgg,
    dropout,
    flatten_layer,
    dense_layer_1,
    prediction_layer
    ])
model.compile(loss= "categorical_crossentropy", optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

print(model.summary())

checkpointer = ModelCheckpoint(filepath = 'vgg16_weights.hdf5', 
                               verbose = False, 
                               save_best_only = True)

early_stop = EarlyStopping(monitor = 'val_loss',
                           patience = 4,
                           restore_best_weights = True,
                           mode = 'min')

result = model.fit(training_data,steps_per_epoch=len(training_data),epochs=50,callbacks=[early_stop, checkpointer], validation_data=testing_data,validation_steps=len(testing_data))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model1.tflite', 'wb') as f:
  f.write(tflite_model)

plt.axis('on')
plt.plot(result.history['accuracy'], label='train acc')
plt.plot(result.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()