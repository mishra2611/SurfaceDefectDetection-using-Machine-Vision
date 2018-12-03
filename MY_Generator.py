
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import keras
import cv2

"""
############## This is generator class to process data in batches and send them for training. ##############
"""
class MY_Generator(keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        arr=[]
        arr2=[]
        for file_name in batch_x:
            img=resize(cv2.imread(file_name, 0),(512,512,1), mode='constant')
            #print(img.shape)
            img = img / 255.0
            arr.append(img)
            img=np.rot90(img)
            arr.append(img)
        for file_name in batch_y:
            img = resize(cv2.imread(file_name, 0), (512,512,1), mode='constant')
            img = img / 255.0
            arr2.append(img)
            img=np.rot90(img)
            arr2.append(img)
        return np.array(arr), np.array(arr2)
