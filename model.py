# Udacity Self-Driving car term 1 project on behavioral neural networks
# contact ytpub@yahoo.com
# last edit 7/25/2017

import csv
import cv2
import numpy as np

# read in steering data
lines = []
with open('../data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

# read out measurements from driving log
# if image is from left or right vantage point, use correction
# to adjust
images = []
measurements = []
correction = 0.2
for line in lines:
  for i in range(3):
    source_path = line[i]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    if i == 1: 
      measurement += correction
    if i == 2:
      measurement -= correction
    measurements.append(measurement)

# add flipped images to data for more balanced training
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
  augmented_images.append(image)
  augmented_measurements.append(measurement)
  augmented_images.append(cv2.flip(image, 1))
  augmented_measurements.append(-measurement)

# set up training data
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Cropping2D

# The following convolutional neural network stack
# is based on the NVIDIA iarchitecture for autonomous vehicles
model = Sequential()
# normalization layer
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
# convolution layers
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
# connected layers
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# adam optimizer instead of gradient descent finds learning rate
# automatically 
model.compile(loss='mse', optimizer='adam')
# run the network on the training data with random shuffling of
# so that training and validation sets are different each time
# it is run.
# Number of epochs and validation set size are set to give a rapidly
# decreasing loss but also to avoid over-fitting
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=2)

model.save('model.h5')
exit()
