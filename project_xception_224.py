import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

import keras as k
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.layers import Input
from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import cv2
from tqdm import tqdm
from heamy.dataset import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score 
from skimage import io,transform

import time

import os
import fnmatch

callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0)]

df_train = pd.read_csv('train_v2.csv')

labels = ['blow_down',
 'bare_ground',
 'conventional_mine',
 'blooming',
 'artisinal_mine',
 'selective_logging',         
 'slash_burn', 
 'cultivation',
 'habitation',
 'road',
 'agriculture',
 'water',
 'primary',
 'partly_cloudy', 
 'cloudy',
 'clear',
 'haze',]

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

np.random.shuffle(df_train.values)

train_values = df_train.values[:36000]
val_values = df_train.values[36000:]

x_train = np.zeros((36000,224,224,3), np.float32)
x_val = np.zeros((40479-36000,224,224,3), np.float32)
y_train = []
y_val = []

i=0

for f, tags in tqdm(train_values, miniters=1000):    
    img = cv2.imread('train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    x_train[i,:,:,:] = np.array(cv2.resize(img, (224, 224)),np.float32)/255.#139 minimum size for inception
    i+=1
    y_train.append(targets)

i=0

for f, tags in tqdm(val_values, miniters=1000):    
    img = cv2.imread('train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    x_val[i,:,:,:] = np.array(cv2.resize(img, (224, 224)),np.float32)/255.#139 minimum size for inception
    i+=1
    y_val.append(targets)
  
y_train = np.array(y_train, np.uint8)
y_val = np.array(y_val, np.uint8)


print(x_train.shape)
print(y_train.shape)

print(x_val.shape)
print(y_val.shape)

train_mean = np.mean(x_train,axis = 0)
x_train -= train_mean
x_val -= train_mean

base_model = Xception(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(17, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

datagen = ImageDataGenerator(
    rotation_range=90,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])    
model.fit_generator(datagen.flow(x_train,y_train, batch_size = 128), validation_data=(x_val, y_val),
                  epochs=10, steps_per_epoch=x_train.shape[0]/ 128, callbacks=callbacks,
                  )

y_pred = model.predict(x_val,batch_size=128)
for thresh in [0.05,0.1,0.15,0.2,0.25,0.3,0.35]:
    print("thresh:",thresh,"\tF2 score:",fbeta_score(y_val, np.array(y_pred)>thresh, beta=2, average='samples'))

for layer in base_model.layers:
    layer.trainable = True
    
#continue with reduced learning rate
model.compile(loss='binary_crossentropy', 
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy']) 
callbacks = [#EarlyStopping(monitor='val_loss', patience=2, verbose=0),
ModelCheckpoint('inc_weights_2.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', 
                verbose=0, save_best_only=True,)]

model.fit_generator(datagen.flow(x_train,y_train, batch_size = 32), validation_data=(x_val, y_val),
                  epochs=10, steps_per_epoch=x_train.shape[0]/ 32, callbacks=callbacks,
                  )

y_pred = model.predict(x_val,batch_size=16)
for thresh in [0.05,0.1,0.15,0.2,0.25,0.3,0.35]:
    print("thresh:",thresh,"\tF2 score:",fbeta_score(y_val, np.array(y_pred)>thresh, beta=2, average='samples'))

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.00005),
              metrics=['accuracy'])

model.fit_generator(datagen.flow(x_train,y_train, batch_size = 32), validation_data=(x_val, y_val),
                  epochs=15, steps_per_epoch=x_train.shape[0]/ 32, callbacks=callbacks,initial_epoch=10
                  )

y_pred = model.predict(x_val,batch_size=16)
for thresh in [0.05,0.1,0.15,0.2,0.25,0.3,0.35]:
    print("thresh:",thresh,"\tF2 score:",fbeta_score(y_val, np.array(y_pred)>thresh, beta=2, average='samples'))

np.save('xception_224_train_mean',train_mean)
