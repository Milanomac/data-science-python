# This project demonstrates the use of image classification model in Keras.
# It was created in IBM course on Keras on Edx platform.
#  
# In this LAB we will use the popular MNIST dataset, a dataset of images.

import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Since we are dealing we images, let's also import the Matplotlib scripting layer in order to view the images.

import matplotlib.pyplot as plt

# importing MNIST dataset from KERAS library.

# import the data
from keras.datasets import mnist
# read the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# checking how many images has the dataset in X_train and X_test
print(X_train.shape)
print(X_test.shape)

plt.imshow(X_train[0])

plt.show()

# With conventional neural networks, we cannot feed in the image as input as is.
# We flatten the images into one-dimensional vectors, each of size 1 x (28 x 28) = 1 x 784:

num_pixels = X_train.shape[1] * X_train.shape[2] # find size of one-dimensional vector
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') # flatten training images
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') # flatten test images

# normalize inputs from 0-255 to 0-1 (because pixel values can range from 0 to 255)
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]
print(num_classes)


# define classification model
def classification_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Train and test the network

# build the model
model = classification_model()

# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)

print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))

# now we can save the pretrained model (accuracy 98%) so that we can invoke it later

model.save('classification_model.h5')

# When you are ready to use your model again, you use the load_model function from keras.models.
# from keras.models import load_model
# pretrained_model = load_model('classification_model.h5')
