#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 13:26:20 2018

@author: saraswatimishra
"""

import numpy as np
import pandas as pd
import cv2
from os import listdir
from os.path import isfile, join
#import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, UpSampling2D
from keras.layers import Lambda, Concatenate
from keras.layers import Conv2DTranspose, Dense, Activation,Conv2D,MaxPooling2D,Flatten,Dropout, InputLayer, Reshape
from keras.optimizers import Adam, Adadelta
from numpy import array
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import random
from tensorflow.python.lib.io import file_io
from io import StringIO
from PIL import Image
import io
import os
import shutil
from skimage.io import imread
from skimage.transform import resize
import  MY_Generator as mv
#from Metrics import Metrics
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from resnet import get_resnet, identity_block, conv_block, up_conv_block
from segnet import get_segnet
from deeplab import get_deeplab
import time

#Implementation of paper, model not used currently
def paper2():
    model=Sequential()
    model.add(InputLayer(input_shape=[512,512,3]))
    model.add(Conv2D(filters=96, kernel_size=(7,7), strides=1,padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=1,padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=1,padding='same', activation='relu'))
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=1,padding='same', activation='relu'))
    model.add(Conv2DTranspose(filters=512, kernel_size=(3,3), activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(units = 1, activation=tf.nn.softmax))
    model.compile(optimizer = 'nadam', loss = 'binary_crossentropy', metrics = ['acc'])
    model.summary()
    return model

#Implementation of a paper, model not used currently
def get_model():
    model = Sequential()  
    model.add(InputLayer(input_shape=[512,512,1]))
    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1,padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=2, padding='same',activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1,padding='same',activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1,padding='same',activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same',activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=2, padding='same',activation='relu'))
    model.add(Conv2D(filters=1024, kernel_size=(3,3), strides=1, padding='same',activation='relu'))
    #model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer = 'adadelta', loss = 'mean_squared_error', metrics = ['accuracy', 'mae'])
    model.summary()
    return model

#model, not used currently
def binary_classifier():
    classifier = Sequential()
    classifier.add(Conv2D(64, kernel_size = (3,3), input_shape = (512,512,3), activation = 'relu',padding='same'))
    classifier.add(Conv2D(64, kernel_size = (3,3), activation = 'relu',padding='same'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Conv2D(64, kernel_size=(3,3), activation = 'relu',padding='same'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dropout(0.3))
    classifier.add(Dense(units = 128, activation = 'relu'))
    #classifier.add(Dense(units = 1, activation=tf.nn.softmax))
    adelta = Adadelta(lr=0.001, rho=0.95, epsilon=None, decay=0.0)
    classifier.compile(optimizer = 'nadam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.summary()
    return classifier

def binary_classifier1():
    classifier = Sequential()
    classifier.add(Conv2D(64, (6, 6), input_shape = (512,512,1), activation = 'relu',padding='same'))
    classifier.add(MaxPooling2D(pool_size = (3, 3)))
    classifier.add(Conv2D(64, (6, 6), activation = 'relu',padding='same'))
    classifier.add(MaxPooling2D(pool_size = (3, 3)))
    classifier.add(Flatten())
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.summary()
    return classifier

#method for calling fit_generator(batch training of images)
def binary_fit(model, training_filenames, GT_training, validation_filenames, GT_validation):
    batch_size=1
    num_training_samples=len(training_filenames)
    num_validation_samples=len(validation_filenames)
    num_epochs=50
    my_training_batch_generator = mv.MY_Generator(training_filenames, GT_training, batch_size)
    my_validation_batch_generator = mv.MY_Generator(validation_filenames, GT_validation, batch_size)
    #metrics = Metrics()
    stmillis = int(round(time.time() * 1000))
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    #print millis
    model.fit_generator(generator=my_training_batch_generator,
                                          steps_per_epoch=(num_training_samples // batch_size),
                                          epochs=num_epochs,
                                          verbose=1,
                                          validation_data=my_validation_batch_generator,
                                          validation_steps=(num_validation_samples // batch_size),
                                          use_multiprocessing=True,
                                          callbacks=[tbCallBack],
                                          workers=5,
                                          max_queue_size=1)
    endmillis = int(round(time.time() * 1000))
    print (endmillis-stmillis)
    model.save('deeplab-1-6.h5')
    model_json = model.to_json()
    with open("deeplab-1-6.json", "w") as json_file:
        json_file.write(model_json)

#def train_model():
    #X_train=get_data("Train")
    #X_train_data = np.array([i[0] for i in X_train]).reshape(-1,512,512,1)
    #Y_train = get_defective_label("Train")
    #Y_train_data = np.array([i[0] for i in Y_train]).reshape(-1,128,128,1)
    #X_test=get_data("Test")
    #X_test_data = np.array([i[0] for i in X_test]).reshape(-1,512,512,1)
    #Y_test = get_defective_label("Test")
    #Y_test_data = np.array([i[0] for i in Y_test]).reshape(-1,128,128,1)
    #model = conv_deconv_model()
    #adelta = Adadelta(lr=0.75, rho=0.95, epsilon=None, decay=0.0)
    #model.compile(optimizer=adelta, loss='mean_squared_error', metrics=['accuracy'])
    #model.summary()
    #model.fit(np.array(X_train_data), np.array(Y_train_data), batch_size=1, epochs=10, validation_data=(X_test_data, Y_test_data))
    

    
#def get_data(imgtype):
    #train_images=[]
    #path="./Class/"
    #for x in range(1,7):
        #path="./Class"+str(x)+"/"
        #prina(path)
        #read_file = file_io.read_file_to_string(path+imgtype+"/Label/Labels.txt")
        #read_file = str(read_file)
        #df = pd.read_fwf(path+imgtype+"/Label/Labels.txt")
        #for i in range(0, len(df)):
           #if(int(df.iloc[i][1])==1):
               #fname=path+imgtype+"/"+str(df.iloc[i][2])
               #print(fname)
               #img=cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
               #img=cv2.resize(img,(512,512))
               #train_images.append([np.array(img)])
    #return train_images
           

def get_class_for_generator(imgtype):
    path="~/Class/"
    filename=[]
    filecode=[]
    filelist={}
    defectmap={}
    fn=[]
    fc=[]
    count=0
    for x in range(1,2):
        path="../Class"+str(x)+"/"
        read_file = file_io.read_file_to_string(path+imgtype+"/Label/Labels.txt")
        read_file = str(read_file)
        df = pd.read_fwf(path+imgtype+"/Label/Labels.txt")
        count=0
        for i in range(0, len(df)):
            imagefile=str(df.iloc[i][2])
            img= cv2.imread(path+imagefile)
            #edges = cv2.Canny(img,100,200)
            #plt.imshow(edges)
            cv2.imwrite(path+"Canny/"+imagefile, img)
            currfile=path+imgtype+"/"+str(df.iloc[i][2])
            if(df.iloc[i][1] == 1):
                filename.append(path+imgtype+"/"+str(df.iloc[i][2]))
                filecode.append(path+imgtype+"/Label/"+str(df.iloc[i][4]))
                filelist[path+imgtype+"/"+str(df.iloc[i][2])]=path+imgtype+"/Label/"+str(df.iloc[i][4])
                defectmap[currfile]=1
            else:
                filename.append(path+imgtype+"/"+str(df.iloc[i][2]))
                fnametest=str(df.iloc[i][2]).split(".")
                filecode.append(gen_black_image(path+imgtype+"/Label/"+fnametest[0]+"_label.PNG"))
                filelist[path+imgtype+"/"+str(df.iloc[i][2])] = str(path+imgtype+"/Label/"+fnametest[0]+"_label.PNG")
                defectmap[currfile]=0
    items=list(filelist.keys())
    shuffle(items)
    for key in items:
        if defectmap[key]==1:
            fn.append(key)
            fc.append(filelist[key])
        elif count<80*6:
            fn.append(key)
            fc.append(filelist[key])
            count=count+1
        print(key, filelist[key])
    return fn, fc
        
def gen_black_image(path):
    img = np.zeros([512,512,1],dtype=np.uint8)
    img.fill(0) 
    cv2.imwrite(path, img)
    return path

#def get_defective_label(imgtype):
    #train_images=[]
    #path="./Class/"
    #for x in range(1,7):
        #path="./Class"+str(x)+"/"
        #print(path)
        #read_file = file_io.read_file_to_string(path+imgtype+"/Label/Labels.txt")
        #read_file = str(read_file)
        #df = pd.read_fwf(path+imgtype+"/Label/Labels.txt")
        #for i in range(0, len(df)):
           #if(int(df.iloc[i][1])==1):
               #fname=path+imgtype+"/"+str(df.iloc[i][2])
               #print(fname)
               #img=cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
               #img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]
               #img=cv2.resize(img,(128,128))
               #train_images.append([np.array(img)])
    #return train_images

def IOU_calc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return 2*(intersection+2) / (K.sum(y_true_f) + K.sum(y_pred_f)+2)

def IOU_calc_loss(y_true, y_pred):
    return -IOU_calc(y_true, y_pred)

def u-net():
    inputs = Input((512,512,1))
    inputs_norm = Lambda(lambda x: x/127.5 - 1.)
    conv1 = Conv2D(8, (3,3), activation = 'relu', padding='same')(inputs)
    conv1 = Conv2D(8, (3,3), activation = 'relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(16, (3,3), activation = 'relu', padding='same')(pool1)
    conv2 = Conv2D(16, (3,3), activation = 'relu', padding='same')(conv2)
    pool2= MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(32, (3,3), activation = 'relu', padding='same')(pool2)
    conv3 = Conv2D(32, (3,3), activation = 'relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(64, (3,3), activation = 'relu', padding='same')(pool3)
    conv4 = Conv2D(64, (3,3), activation = 'relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

    conv5 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool4)
    conv5 = Conv2D(128, (3,3), activation = 'relu', padding='same')(conv5)

    mid1 = UpSampling2D(size=(2,2))(conv5)
    up6 = keras.layers.concatenate([mid1, conv4], axis=3)
    conv6 = Conv2D(64, (3,3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(64, (3,3), activation='relu', padding='same')(conv6)

    mid2 = UpSampling2D(size=(2,2))(conv6)
    up7 = keras.layers.concatenate([mid2, conv3], axis=3)
    conv7 = Conv2D(32, (3,3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(32, (3,3), activation='relu', padding='same')(conv7)

    mid3 = UpSampling2D(size=(2,2))(conv7)
    up8 = keras.layers.concatenate([mid3, conv2], axis=3)
    conv8 = Conv2D(16, (3,3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(16, (3,3), activation='relu', padding='same')(conv8)

    mid4 = UpSampling2D(size=(2,2))(conv8)
    up9 = keras.layers.concatenate([mid4, conv1], axis=3)
    conv9 = Conv2D(8, (3,3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(8, (3,3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1,1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs = conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss=IOU_calc_loss, metrics=[IOU_calc])
    model.summary()
    return model



#train_model();
#binary_classifier();
X_train, Y_train = get_class_for_generator("Train")
X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.30)
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
#X_test, Y_test = get_class_for_generator("Test")  
#binary_model = u-net()
#binary_model = get_resnet(f=16, bn_axis=3, classes=1) 
binary_model=get_deeplab()
#binary_model.save('binary-model-1.h5')
binary_fit(binary_model, X_train, y_train, X_test, y_test) 

