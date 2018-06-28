'''
The MNIST database of handwritten digits, has a training set of 60,000 examples,
and a test set of 10,000 examples. It is a subset of a larger set available from NIST.
The digits have been size-normalized and centered in a fixed-size image. 
This classifier would help to categorize different handwritten images to
the specified set of 10 classes, using keras library based machine learning algorithm.
'''
import numpy as np
from datetime import datetime
from keras.datasets import mnist
import matplotlib.pyplot as plt

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


seed = 9
np.random.seed(seed)


# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_validate = X_test[8:12]
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')


def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# build the model
model = larger_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))



(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_validate = X_test[0:4]

for i in range(4):
 plt.subplot(2, 2, i+1)
 plt.imshow(X_validate[i], cmap='gray')
 plt.axis('off')

X_validate = X_validate.reshape(X_validate.shape[0], 1, 28, 28).astype('float32')
model.predict(X_validate)
