import numpy as np
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd

# https://www.kaggle.com/datamunge/sign-language-mnist
train_data=pd.read_csv('sign_mnist_test.csv')
print(train_data.shape)

train_data_splitted=np.hsplit(train_data,[1])
labels=train_data_splitted[0].to_numpy()
images=train_data_splitted[1].to_numpy().reshape(labels.size,28,28)

print(np.unique(labels).size)

print(labels.shape)
print(images.shape)

images = np.expand_dims(images, axis=3)
labels = np.expand_dims(labels, axis=3)

# x = np.arange(40.0).reshape(8, 5)
# print(x)
# sx=np.hsplit(x,[1])
# print(np.unique(sx[0]).size)
# print(sx[1].reshape(8,2,2))

training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


trd=training_datagen.flow(
    images,
    labels,

    batch_size=28
)
print('ok')