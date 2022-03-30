from keras.layers import Conv2D, UpSampling2D
from keras.layers import InputLayer
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
import numpy as np
import os
import cv2


def mse(imageA, imageB):
    err = np.sum((imageA.astype('float') - imageB.astype('float')) ** 2)
    print(imageA.shape, imageB.shape)
    err /= float(imageA.shape[0] * imageA.shape[1])
    print('mse:', err)


original = cv2.imread('org.jpg')
contrast = cv2.imread('2000 epok.png')
original = np.array(original, dtype=float)
contrast = np.array(contrast, dtype=float)
print(original.shape)
mse(original, contrast)