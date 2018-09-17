#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 21:27:53 2018

@author: saraswatimishra
"""

import numpy as np
import pandas as pd
import cv2
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
onlyfiles = [f for f in listdir("./Documents/Class1/Train/") if isfile(join("./Documents/Class1/Train/", f))]
images = []
print(onlyfiles)
#for x in range(len(onlyfiles)):
#    images[x] = read_image('./Documents/Class1/Train/'+onlyfiles[x])
onlyfiles.remove('Thumbs.db')
onlyfiles.remove('.DS_Store')
TRAIN_PATH='./Documents/Class1/Train/'
nparray=[]
images = np.empty(len(onlyfiles), dtype=object)
edges_canny = np.empty(len(onlyfiles), dtype=object)
edges_laplacian = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    filename_canny = './Documents/Class1/Train/Edge_Canny/'+onlyfiles[n].split('.')[0]+'_edge_canny.png'
    filename_laplacian = './Documents/Class1/Train/Edge_Laplacian/'+onlyfiles[n].split('.')[0]+'_edge_laplacian.png'
    #print(filename)
    img= cv2.imread('./Documents/Class1/Train/'+onlyfiles[n])
    #img_lap= cv2.imread('./Documents/Class1/Train/'+onlyfiles[n])
    edges_canny[n] = cv2.Canny(img, 100, 200)
    
    cv2.imwrite(filename_canny, edges_canny[n])
    
    #img = np.array(img, np.float32) / 255
    #edges_canny[n] = np.array(edges_canny[n], np.float32) / 255
    images[n] = img
    #Applying Laplacian Filter
    denoised = cv2.GaussianBlur(img,(5,5),0)
    edges_laplacian[n] = cv2.Laplacian(img, cv2.CV_32F)
    cv2.imwrite(filename_laplacian, edges_laplacian[n])
    
    