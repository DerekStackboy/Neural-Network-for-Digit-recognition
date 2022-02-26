import tensorflow as tf
import cv2
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from keras.datasets import mnist
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#Loading MINSIT database 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#standardize the value so that they are in the range from 0 to 1
x_train = x_train / 255
x_test = x_test / 255

#y_train values that are output like numbers will be presented like a vector of length 10
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

#add another dimension (channel)
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

#sequntial -- creating a multilayer neural network model
model = keras.Sequential([
    #first convolutional layer (32 filters (3 3 cores))
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28,28,1)),
    #the next layer enlarges the scale of the obtained features, stridens = 2(scan step = 2)
    MaxPooling2D((2,2), strides=2),
    #another convolutional layer that uses the previous feature map and analyzes with a field of 64 filters
    Conv2D(64, (3,3), padding='same', activation='relu' ),
    MaxPooling2D((2,2), strides=2),
    #flatten converts the input matrix into a layer consisting of a vector
    Flatten(),
    #connects neurons to layers
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
    
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics =['accuracy'])

#starting the learning process (input training set, desired value at the output of the neural network,
#batch size (after every 32 images we will adjust the weight values),
#validation_split(splits the training sample into training-80% and testing-20%))
his = model.fit(x_train, y_train_cat, batch_size=32, epochs = 7, validation_split=0.2)

#model.evaluate(x_test, y_test_cat)

def testing():
    img = cv2.imread('image.png', 0)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0

    #here network guessing our handwritten number 
    pred = model.predict(img)
    return pred

    