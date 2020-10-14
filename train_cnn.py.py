import numpy as np
import pickle
import cv2, os
from glob import glob
from keras import optimizers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import  np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as Back 

Back.set_image_data_format('channels_last')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def image_size():
  img = cv2.imread('gestures/1/100.jpg', 0)
  return img.shape

#print(image_size())

def num_of_classes():
  return len(glob('gestures/*'))

#print(num_of_classes())

img_x, img_y = image_size()

def cnn_model():
  num_classes = num_of_classes()
  model = Sequential()
  model.add(Conv2D(16, (2,2), input_shape= (img_x, img_y, 1), activation= 'relu'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
  model.add(Conv2D(32, (3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(3,3), strides=(3,3), padding='same'))
  model.add(Conv2D(64, (5,5), activation='relu'))
  model.add(MaxPooling2D(pool_size= (5,5), strides=(5,5), padding='same'))
  model.add(Flatten())
  model.add(Dense(128, activation= 'relu'))
  model.add(Dropout(0.2))
  model.add(Dense(num_classes, activation='softmax'))
  sgd = optimizers.SGD(learning_rate=1e-2)
  model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
  filepath= "cnn_model_keras2.h5"
  checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
  callback_list = [checkpoint1]

  return model, callback_list

def training():
  with open("train_images", "rb") as f:
    train_images = np.array(pickle.load(f))
  with open("train_labels", "rb") as f:
    train_labels = np.array(pickle.load(f), dtype=np.int32)
  
  with open("val_images", "rb") as f:
    val_images = np.array(pickle.load(f))
  with open("val_labels", "rb") as f:
    val_labels = np.array(pickle.load(f), dtype=np.int32)
  
  train_images = np.reshape(train_images, (train_images.shape[0], img_x, img_y, 1))
  val_images = np.reshape(val_images, (val_images.shape[0], img_x, img_y, 1))
  train_labels = np_utils.to_categorical(train_labels)
  val_labels = np_utils.to_categorical(val_labels)

  print(val_labels.shape)

  model, callbacks_list = cnn_model()
  model.summary()
  model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=15, batch_size=500, callbacks= callbacks_list)
  print('\n')
  scores = model.evaluate(val_images, val_labels, verbose=0)
  print("CNN Error : %.2f%%" %(100-scores[1]*100))
  #model.save('cnn_model_keras2.h5')


training()
Back.clear_session();
