#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 13:26:20 2018

@author: saraswatimishra
"""
"""
All the imports required to run the file are mentioned here
"""
import numpy as np
import pandas as pd
import cv2
from os import listdir
from os.path import isfile, join
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
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from resnet import get_resnet, identity_block, conv_block, up_conv_block
from segnet import get_segnet
from deeplab import get_deeplab
from unet import get_unet
import time
import sys
from metrics import dice_coef, defect_accuracy, precision, recall, f1score, dice_coef_loss
from customCallback import Histories


"""
 ############## Function for calling fit_generator to batch train the model ###############
"""
def binary_fit(model, training_filenames, GT_training, validation_filenames, GT_validation, modelname):
    batch_size=1
    num_training_samples=len(training_filenames)
    num_validation_samples=len(validation_filenames)
    num_epochs=50
    my_training_batch_generator = mv.MY_Generator(training_filenames, GT_training, batch_size)
    my_validation_batch_generator = mv.MY_Generator(validation_filenames, GT_validation, batch_size)
    stmillis = int(round(time.time() * 1000))
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph-UNET', histogram_freq=0, write_graph=True, write_images=True)
    #histories = Histories(my_validation_batch_generator, my_training_batch_generator)
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
    model.save(modelname+"_acc.h5")
    model_json = model.to_json()
    with open(modelname+"_acc.json", "w") as json_file:
        json_file.write(model_json)
           


"""
################ Read data as batches and send them to the model ############################
"""
def get_class_for_generator(imgtype):
    path="~/Class/"
    filename=[]
    filecode=[]
    filelist={}
    defectmap={}
    fn=[]
    fc=[]
    count=0
    numclasses=6
    for x in range(1,numclasses+1):
        path="../Class"+str(x)+"/"
        read_file = file_io.read_file_to_string(path+imgtype+"/Label/Labels.txt")
        read_file = str(read_file)
        df = pd.read_fwf(path+imgtype+"/Label/Labels.txt")
        count=0
        for i in range(0, len(df)):
            imagefile=str(df.iloc[i][2])
            img= cv2.imread(path+imagefile)
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
        elif count<80*numclasses:
            fn.append(key)
            fc.append(filelist[key])
            count=count+1
        print(key, filelist[key])
    return fn, fc
  

"""
################## generate black images for non defective images ########################
"""      
def gen_black_image(path):
    img = np.zeros([512,512,1],dtype=np.uint8)
    img.fill(0) 
    cv2.imwrite(path, img)
    return path



"""
################ The main code flow to call approriate functions ##########################
"""
if len(sys.argv) < 2:
    sys.exit(0)
cmd=sys.argv[1]
if cmd == "1":
    binary_model = get_unet(n_filters=16, dropout=0.05, batchnorm=True)
    modelname="unet"
elif cmd == "2":
    binary_model = get_resnet(f=16, bn_axis=3, classes=1)
    modelname="resnet"
elif cmd == "3":
    binary_model = get_segnet()
    modelname="segnet"
else:
    binary_model=get_deeplab()
    modelname="deeplab"
X_train, Y_train = get_class_for_generator("Train")
X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.30)
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
binary_fit(binary_model, X_train, y_train, X_test, y_test, modelname) 

