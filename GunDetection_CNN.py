'''
Project : MAE 551
Gun Detection using Convolutional Neural Network

Project team:
Deep Patel
Zhiming Zuang
Erik

'''

'''Importing the Libraries'''
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.metrics import metrics
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Activation, BatchNormalization
from keras import optimizers
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import os
import cv2
import matplotlib.pyplot as plt



'''Importing the Data'''
# Here dataset is created by importing the images using the iterator "ImageDataGenerator" from tensorflow.keras.
# Data is imported in two classes named "gun_found" and "gun_not_found" and images are preprocessed using vgg16 engine and converted to greyscale.
# Entire data set of more than 419000 files is split into training, validation and test sets.


datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=tf.keras.applications.vgg16.preprocess_input)

path_train_data = './Data/train'
X_train = datagen.flow_from_directory(directory=path_train_data, target_size=(100,100),classes=['gun_found','no_gun_found'], batch_size=50, shuffle=True)

path_valid_data = './Data/valid'
X_valid = datagen.flow_from_directory(directory=path_valid_data, target_size=(100,100),classes=['gun_found','no_gun_found'], batch_size=50, shuffle=True)

path_test_data = './Data/test'
X_test = datagen.flow_from_directory(directory=path_test_data, target_size=(100,100), classes=['gun_found','no_gun_found'], batch_size=50, shuffle=True)


''' Function for Visualization of the Data '''
# This function is used to visualize the imported images and sanity check for the image preprocessing.


def dispImage(array):
  fig1, ax = plt.subplots(10,5, figsize=(100,100))
  ax = ax.flatten()
  for img, ax in zip(array, ax):
    ax.imshow(img)
    ax.axis('off')
  plt.tight_layout()
  plt.show()


a,b = next(X_train)
dispImage(a)
print(b)

''' Machine Learning Model - CNN'''
# Convolutional Neural Network (CNN) is created sunig keras API by Tensorflow.
# CNN consists 4 convolutional layers, each followed by a maxpooling layer.
# CNN layer is followed by the dense layers, creating a Fully Connected Neural Network.



# Creating the Model
model = Sequential(name='gun_CNN_classifier')

# Set of Layer-1 (Convolutional + MaxPooling)

model.add(Conv2D(64, (3,3), activation='relu', padding = 'same', input_shape=(100,100,3),kernel_initializer='random_normal', name='conv1'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, name='maxpool1'))
          
# Set of Layer-2 (Convolutional + MaxPooling)

model.add(Conv2D(128, (3,3), activation='relu', padding = 'same', name='conv2'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, name='maxpool2'))

# Set of Layer-3 (Convolutional + MaxPooling)

model.add(Conv2D(256, (3,3), activation='relu', padding = 'same', name='conv3'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, name='maxpool3'))

# Set of Layer-4 (Convolutional + MaxPooling)

model.add(Conv2D(512, (3,3), activation='relu', padding='same', name='conv4'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, name='maxpool4'))


# Flattening
model.add(Flatten(name='flatten'))

# Fully Connected Neural Network
model.add(Dense(64, activation='relu',  kernel_regularizer=regularizers.L2(4e-3), name='dense1')) # To avoid overfitting, regularization parameter of value 4x10^-3 is used in dense layers.
model.add(Dense(128, activation='relu',  kernel_regularizer=regularizers.L2(4e-3), name='dense2'))
model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.L2(4e-3), name='Output'))

# Compilation of Model
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["mae","acc"])
          
model.summary() # Prints the Model Summary

# fitting the model
ep=1
GD = model.fit(x=X_train, validation_data=X_valid, epochs=ep, verbose=1)
scores = model.evaluate(x=X_valid)
print(scores)

''' Saving the Model '''

filepath = "./Model/"

model.save( filepath, overwrite=True, include_optimizer=True, save_traces=True)

''' Analyzing the performance of the Model '''

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,ep),GD.history['acc'], Label='Accuracy - Training Data')
plt.plot(np.arange(0,ep),GD.history['val_acc'], Label='Accuracy - Validation Data')
plt.plot(np.arange(0,ep),GD.history['loss'], Label='Loss - Training Data')
plt.plot(np.arange(0,ep),GD.history['val_loss'], Label='Loss - Validation Data')
plt.legend(['Accuracy - Train','Accuracy - Valid','Loss - train','Loss - Valid'])
plt.title('Loss and Accuracy over Datasets')
plt.xlabel('Epoch')
plt.ylabel('Loss & Accuracy')
plt.show()