#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:10:31 2018

@author: logicalsteps
"""
#import opencv
import os
import cv2
import numpy as np
from datetime import datetime
from keras.datasets import mnist
import matplotlib.pyplot as plt
from random import shuffle
from skimage.transform import resize

import numpy as np
from skimage.transform import resize
from skimage import color
from scipy import misc
import glob

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')

#############################Loading the Train images###########################################
def load_images_from_folder(folder):
    images = []
    label = []
    for filename in os.listdir(folder):
        lbl = filename.split('.')[0]
        img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
        if lbl is not None:
            label.append(lbl)
    return images, label

#os.chdir("C:/Users/LS/Downloads/Image Classification/Flower Classification Average Grey")
#Train_path = os.getcwd()
Train_path = "Flower Classification Average Grey"
images,label = load_images_from_folder(Train_path)

#Displaying image
cv2.imshow('image',images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

trainY = np.empty([len(label),2])
for i in range(0,len(label)):
    if label[i].startswith('Flower'):
        trainY[i] = [1,0]
    elif label[i].startswith('NonFlower'):
        trainY[i] = [0,1]

#Normalising numpy image array
images_N = []
for i in range(0,len(images)):
    nl = (images[i].astype('float32'))/255     
    images_N.append(nl)
    
images_N = np.array(images_N)
images_N = images_N.reshape(images_N.shape[0], 1, images_N.shape[1], images_N.shape[2])

seed = 9
np.random.seed(seed)


#############################Training the images###########################################
def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 282, 396), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(2, activation='softmax'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# build the model
model = larger_model()
# Fit the model
model.fit(images_N, trainY, epochs=8, batch_size=10)


#############################Loading the Test images###########################################
#os.chdir('C:/Users/LS/Downloads/Image Classification/TestData')
#Test_path = os.getcwd()
Test_path = "TestData"
imagesTest = []
labelTest = []
for filename in os.listdir(Test_path):
    img = cv2.imread(os.path.join(testPath,filename),cv2.IMREAD_GRAYSCALE)
    lbl = filename.split('.')[0]
    if img is not None:
        img=resize(img, (282, 396), mode='constant')
        imagesTest.append(img)
        if lbl is not None:
            labelTest.append(lbl) 

testY = np.empty([len(labelTest),2])
for i in range(0,len(labelTest)):
    if labelTest[i].startswith('Flower'):
        testY[i] = [1,0]
    elif labelTest[i].startswith('NonFlower'):
        testY[i] = [0,1]

    
imagesTest = np.array(imagesTest)
imagesTest = imagesTest.reshape(imagesTest.shape[0], 1, imagesTest.shape[1], imagesTest.shape[2])
imagesTest.shape


#Evaluation and Prediction
scores = model.evaluate(imagesTest, testY, verbose=0)
predictY = model.predict(imagesTest)




