from keras.layers import Conv2D, UpSampling2D
from keras.layers import InputLayer
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import cv2


def mse(imageA, imageB):
    err = np.sum((imageA.astype('float') - imageB.astype('float')) ** 2)
    print(imageA.shape, imageB.shape)
    err /= float(imageA.shape[0] * imageA.shape[1])
    print('mse:', err)


image = img_to_array(load_img('test_image1.jpg'))
image = np.array(image, dtype=float)

X = rgb2lab(1.0 / 255 * image)[:, :, 0]
Y = rgb2lab(1.0 / 255 * image)[:, :, 1:]
Y /= 128
X = X.reshape(1, 128, 128, 1)
Y = Y.reshape(1, 128, 128, 2)

image2 = img_to_array(load_img('test_image4.png'))
image2 = np.array(image2, dtype=float)

X2 = rgb2lab(1.0 / 255 * image2)[:, :, 0]
Y2 = rgb2lab(1.0 / 255 * image2)[:, :, 1:]
Y2 /= 128
X2 = X2.reshape(1, 128, 128, 1)
Y2 = Y2.reshape(1, 128, 128, 2)

model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

model.compile(optimizer='rmsprop', loss='mse')

model.fit(x=X,
          y=Y,
          batch_size=1,
          epochs=100)

output = model.predict(X2)
output *= 128

cur = np.zeros((128, 128, 3))

cur[:, :, 0] = X2[0][:, :, 0]
cur[:, :, 1:] = output[0]

imsave("img_result_lab.jpg", (cur))
imsave("img_result.jpg", lab2rgb(cur))
imsave("img_gray_version.jpg", rgb2gray(lab2rgb(cur)))

original = cv2.imread('test_image3.jpg')
contrast = cv2.imread('img_result.jpg')
original = np.array(original, dtype=float)
contrast = np.array(contrast, dtype=float)

mse(original, contrast)