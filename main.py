from keras.layers import Conv2D, UpSampling2D
from keras.layers import InputLayer
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
import numpy as np
import os


X = []
for filename in os.listdir('../Projekt - Kolorowanie obrazów/train/'):
    X.append(img_to_array(load_img('../Projekt - Kolorowanie obrazów/train/' + filename)))
X = np.array(X, dtype=float)

split = int(0.95 * len(X))
Xtrain = X[:split]
Xtrain = 1.0 / 255 * Xtrain

model = Sequential()
model.add(InputLayer(input_shape=(128, 128, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])


datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    horizontal_flip=True)

batch_size = 10


def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:, :, :, 0]
        Y_batch = lab_batch[:, :, :, 1:] / 128
        yield (X_batch.reshape(X_batch.shape + (1,)), Y_batch)


Xtest = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 0]
Xtest = Xtest.reshape(Xtest.shape + (1,))
Ytest = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 1:]
Ytest = Ytest / 128

tensorboard = TensorBoard(log_dir="output/first_run")
model.fit_generator(image_a_b_gen(batch_size), callbacks=[tensorboard], epochs=100, steps_per_epoch=10, validation_data=(Xtest,Ytest))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

Xtest = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 0]
Xtest = Xtest.reshape(Xtest.shape + (1,))
Ytest = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 1:]
Ytest = Ytest / 128
print(model.evaluate(Xtest, Ytest, batch_size=batch_size))

color_me = []
for filename in os.listdir('../Projekt - Kolorowanie obrazów/test/'):
    color_me.append(img_to_array(load_img('../Projekt - Kolorowanie obrazów/test/' + filename)))
color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0 / 255 * color_me)[:, :, :, 0]
color_me = color_me.reshape(color_me.shape + (1,))

model.save('model_save')

output = model.predict(color_me)
output = output * 128

for i in range(len(output)):
    cur = np.zeros((128, 128, 3))
    cur[:, :, 0] = color_me[i][:, :, 0]
    cur[:, :, 1:] = output[i]
    imsave("result/img_" + str(i) + ".png", lab2rgb(cur))
