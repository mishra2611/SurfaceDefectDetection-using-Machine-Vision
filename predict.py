import numpy as np
import pandas as pd
import cv2
from os import listdir
from os.path import isfile, join
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, UpSampling2D
from keras.layers import merge
from keras import backend as K
from keras.layers import Lambda, Concatenate
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
from keras.models import load_model, Model
from keras import backend as K
import io
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib
from keras.utils import plot_model
from skimage.transform import resize
from metrics import dice_coef, dice_coef_loss, precision, recall, f1score
from os import listdir
import sys

def get_file_from_custom_folder(path):
    train_images=[]
    filenames=[]
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for f in files:
        currfile=path+f
        print(currfile)
        img=cv2.imread(currfile, cv2.IMREAD_GRAYSCALE)
        img=resize(img,(512,512,1))
        train_images.append(img)
        fname=f.split(".")
        filenames.append(fname[0])
    return train_images, filenames



# def get_data_from_test_folder(imgtype):
#     train_images=[]
#     filenames=[]
#     path="~/Documents/Class/"
#     for x in range(7,8):
#         path="./Class"+str(x)+"/"
#         #prina(path)
#         read_file = file_io.read_file_to_string(path+imgtype+"/Label/Labels.txt")
#         read_file = str(read_file)
#         df = pd.read_fwf(path+imgtype+"/Label/Labels.txt")
#         for i in range(0, len(df)):
#            if(int(df.iloc[i][1])==1):
#                fname=path+imgtype+"/"+str(df.iloc[i][2])
#                print(fname)
#                img=cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
#                img=resize(img,(512,512,1))
#                train_images.append(img)
#                f=df.iloc[i][2].split(".")
#                filenames.append(f[0])
#                print(f[0])
#     return train_images, filenames

def get_file_from_custom_folder_contour(path):
    train_images=[]
    filenames=[]
    copyimages = []
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for f in files:
        currfile=path+f
        print(currfile)
        img=cv2.imread(currfile, cv2.COLOR_BGR2GRAY)
        train_images.append(img)
        fname=f.split(".")
        filenames.append(fname[0])
    return train_images, filenames


def predict_images(path, model):
    model = load_model(model,custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef, 'precision':precision, 'recall':recall, 'f1score':f1score})
    X_train, filenames=get_file_from_custom_folder(path+"/")
    X_train_data = np.array(X_train)
    predicted_mask_batch = model.predict(X_train_data)
    predicted_mask_batch = predicted_mask_batch*255
    for x in range(len(predicted_mask_batch)):
        cv2.imwrite(path+"/predictedImages/"+filenames[x]+".jpg", predicted_mask_batch[x])  
    

def countour_images(path):
    origpath=path
    path=path+"/predictedImages/"
    X_train_data, filenames=get_file_from_custom_folder_contour(path)
    for i in range(len(X_train_data)):
        (thresh, im_bw) = cv2.threshold(X_train_data[i], 127, 255, cv2.THRESH_BINARY )
        _,contours,hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
        #img_data = cv2.drawContours(X_train_data[x], contours, -1, (0,255,0), 3)
            img2 = cv2.imread(origpath+"/"+filenames[i]+".PNG", cv2.COLOR_BGR2GRAY)
            img3 = X_train_data[i].copy()
            for t in range(len(contours)):
                if cv2.contourArea(contours[t]) < 100:
                    #print("Image"+filenames[i]+"is not defective")
                    continue

                print("Image "+filenames[i]+"is defective")
                x,y,w,h = cv2.boundingRect(contours[t])
                #cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),3)
                print("The location (x,y, width, height) of defect is: "+str(x)+","+" "+str(y)+","+str(w)+","+str(h))
            #cv2.imwrite("/Users/saraswatimishra/Downloads/test/SurfaceDefectDetection-using-Machine-Vision/contourImages/"+filenames[i]+".jpg", img2)
            #cv2.imwrite("/Users/saraswatimishra/Downloads/test/SurfaceDefectDetection-using-Machine-Vision/contourImages/"+filenames[i]+"_og.jpg", ogImages[i])
            cv2.imwrite(path+filenames[i]+"_predicted.jpg", img2)
        else:
            print("Image "+filenames[i]+" is non-defective")





np.set_printoptions(threshold=np.inf, precision=8, floatmode='maxprec')
path=sys.argv[1]
model=sys.argv[2]
predict_images(path, model)
countour_images(path)


