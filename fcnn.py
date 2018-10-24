import numpy as np
import pandas as pd
import cv2
from os import listdir
from os.path import isfile, join
#import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Conv2D,MaxPooling2D,Flatten,Dropout, InputLayer, Reshape
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

def get_model():
    model = Sequential()  
    model.add(InputLayer(input_shape=[512,512,1]))
    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1,data_format='channels_last',padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1,data_format='channels_last', padding='same',activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1,data_format='channels_last', padding='same',activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1,data_format='channels_last', padding='same',activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=1,data_format='channels_last', padding='same',activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=1,data_format='channels_last', padding='same',activation='relu'))
    model.add(Conv2D(filters=1024, kernel_size=(3,3), strides=1,data_format='channels_last', padding='same',activation='relu'))
    #model.add(Flatten())
    model.add(Dense(input_dim=[128,128,1], units=1))
    return model

def train_model():
    X_train=get_data("Train")
    X_train_data = np.array([i[0] for i in X_train]).reshape(-1,512,512,1)
    Y_train = get_defective_label("Train")
    Y_train_data = np.array([i[0] for i in Y_train]).reshape(-1,512,512,1)
    model = get_model()
    adelta = Adadelta(lr=0.75, rho=0.95, epsilon=None, decay=0.0)
    model.compile(optimizer=adelta, loss='mean_squared_error', metrics=['accuracy'])
    model.summary()
    model.fit(np.array(X_train_data), np.array(Y_train_data), batch_size=1, epochs=10)
    
    
def get_data(imgtype):
    train_images=[]
    path="./Class/"
    for x in range(1,7):
        path="./Class"+str(x)+"/"
        print(path)
        read_file = file_io.read_file_to_string(path+imgtype+"/Label/Labels.txt")
        read_file = str(read_file)
        df = pd.read_fwf(path+imgtype+"/Label/Labels.txt")
        for i in range(0, len(df)):
           if(int(df.iloc[i][1])==1):
               fname=path+imgtype+"/"+str(df.iloc[i][2])
               print(fname)
               img=cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
               img=cv2.resize(img,(512,512))
               train_images.append([np.array(img)])
    return train_images
           


def get_defective_label(imgtype):
    train_images=[]
    path="./Class/"
    for x in range(1,7):
        path="./Class"+str(x)+"/"
        print(path)
        read_file = file_io.read_file_to_string(path+imgtype+"/Label/Labels.txt")
        read_file = str(read_file)
        df = pd.read_fwf(path+imgtype+"/Label/Labels.txt")
        for i in range(0, len(df)):
           if(int(df.iloc[i][1])==1):
               fname=path+imgtype+"/"+str(df.iloc[i][2])
               print(fname)
               img=cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
               img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]
               img=cv2.resize(img,(512,512))
               train_images.append([np.array(img)])
    return train_images


train_model();
