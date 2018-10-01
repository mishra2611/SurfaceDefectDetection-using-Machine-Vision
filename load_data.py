#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 20:08:59 2018

@author: saraswatimishra
"""

import numpy as np
import pandas as pd
import cv2
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Conv2D,MaxPooling2D,Flatten,Dropout, InputLayer
from keras.optimizers import Adam
from numpy import array
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import random

onlyfiles = [f for f in listdir("./Documents/Class1/Train/") if isfile(join("./Documents/Class1/Train/", f))]
onlyfiles.remove('Thumbs.db')
onlyfiles.remove('.DS_Store')
onlyfiles_test = [f for f in listdir("./Documents/Class1/Test/") if isfile(join("./Documents/Class1/Test/", f))]
onlyfiles_test.remove('Thumbs.db')
onlyfiles_test.remove('.DS_Store')
test_labels=np.genfromtxt('/Users/saraswatimishra/Documents/Class1/Test/Label/Labels.txt', usecols=(0,1))
test_defects = {}
for x in test_labels:
    test_defects[str(x[0])] = str(x[1])

test_labels=np.genfromtxt('/Users/saraswatimishra/Documents/Class1/Train/Label/Labels.txt', usecols=(0,1))
map_of_defects = {}
for x in test_labels:
    map_of_defects[str(x[0])] = str(x[1])

x_train=get_training_data()
x_train_test = get_testing_data()
x_train_train = x_train[:]
random.shuffle(x_train)

x_train_data = np.array([i[0] for i in x_train_train]).reshape(-1,512,512,1)
y_train_data = np.array([i[1] for i in x_train_train])
x_test_data = np.array([i[0] for i in x_train_test]).reshape(-1,512,512,1)
y_test_data = np.array([i[1] for i in x_train_test])

model = Sequential()  
model.add(InputLayer(input_shape=[512,512,1]))
model.add(Conv2D(filters=32, kernel_size=3, strides=1,padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same',activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=1,padding='same',activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=1,padding='same',activation='relu'))
model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding='same',activation='relu'))
model.add(Conv2D(filters=512, kernel_size=3, strides=2, padding='same',activation='relu'))
model.add(Conv2D(filters=1024, kernel_size=3, strides=1, padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=3, padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(2, activation='softmax'))
sgd = Adam(lr=0.75)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(x_train_data, y_train_data, batch_size=100, epochs=10, validation_data=(x_test_data, y_test_data))
model.summary()

def get_training_data():
    path='./Documents/Class1/Train/Edge_Canny/'
    train_images=[]
    for n in range (0, len(onlyfiles)):
        img= cv2.imread('./Documents/Class1/Train/'+onlyfiles[n], cv2.IMREAD_GRAYSCALE)
        fname = str(int(onlyfiles[n].split('.')[0])) +".0"
        img=cv2.resize(img,(512,512))
        tmp=img
        defect = int(float(map_of_defects[fname]))
        defect_value = [0,0]
        if(defect == 1):
            defect_value[0]=1
        else:
            defect_value[1]=0
        train_images.append([np.array(tmp), defect_value])
    return train_images



def get_testing_data():
    path='./Documents/Class1/Train/Edge_Canny/'
    train_images=[]
    for n in range (0, len(onlyfiles)):
        img= cv2.imread('./Documents/Class1/Test/'+onlyfiles_test[n], cv2.IMREAD_GRAYSCALE)
        fname = str(int(onlyfiles_test[n].split('.')[0])) +".0"
        img=cv2.resize(img,(512,512))
        tmp=img
        defect = int(float(test_defects[fname]))
        defect_value = [0,0]
        if(defect == 1):
            defect_value[0]=1
        else:
            defect_value[1]=0
        train_images.append([np.array(tmp), defect_value])
    return train_images