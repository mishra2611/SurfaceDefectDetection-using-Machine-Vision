import cv2 # used for resize. if you dont have it, use anything else
import numpy as np
from model import Deeplabv3
from model import relu6, BilinearUpsampling
from metrics import dice_coef, dice_coef_loss, precision, recall, f1score
from keras.optimizers import Adam, Adadelta

def get_deeplab():
    deeplab_model = Deeplabv3(input_shape=(512,512,1), weights=None,classes=1, OS=16)
    deeplab_model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss,
                  metrics=[dice_coef, 'accuracy', precision, recall])
    #deeplab_model.load_weights('deeplabv3_weights_tf_dim_ordering_tf_kernels.h5', by_name = True)
    deeplab_model.summary()
    return deeplab_model



