
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import keras
import cv2

class MY_Generator(keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        arr = np.array([resize(cv2.imread(file_name), (512,512,1), mode='constant') for file_name in batch_x])
        arr2 = np.array([resize(cv2.imread(file_name), (512,512,1), mode='constant') for file_name in batch_y])
        #arr2 = np.array(batch_y)
        #arr = arr.astype("float")/255.0
        #arr2 = arr2.astype("float")/255.0
        return arr,arr2
