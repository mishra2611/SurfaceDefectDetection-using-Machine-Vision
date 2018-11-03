import numpy as np
import pandas as pd
import cv2
from os import listdir
from os.path import isfile, join
#import matplotlib.pyplot as plt
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
from keras.models import load_model
import io
import os
import shutil


def IOU_calc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return 2*(intersection+1) / (K.sum(y_true_f) + K.sum(y_pred_f) +1)

def IOU_calc_loss(y_true, y_pred):
    return -IOU_calc(y_true, y_pred)


np.set_printoptions(threshold=np.inf)
model = load_model('loc1.h5',custom_objects={'IOU_calc_loss': IOU_calc_loss, 'IOU_calc': IOU_calc})
fname="./Class1/Train/0616.PNG"
img=cv2.imread(fname, cv2.IMREAD_GRAYSCALE).reshape(-1,512,512,1)
#img=cv2.resize(img,(512,512))
print(img)
result = model.predict(img, batch_size=1)
#print(result)

