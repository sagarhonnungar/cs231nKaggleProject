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

thresh_2_val = np.load('xception_224_thresh_2_val.npy')
train_mean = np.load('xception_224_train_mean.npy')

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

df_test = pd.read_csv('sample_submission_v2.csv')

model.load_weights('xception_224_weights_2')

x_test = []
x_test = np.zeros((30000,224,224,3), np.float32)


i = 0 
for f, tags in tqdm(df_test.values[:30000], miniters=1000):
    img = cv2.imread('test-jpg/{}.jpg'.format(f))
    x_test[i,:,:,:] = np.array(cv2.resize(img, (224, 224)),np.float32)/255.#139 minimum size for inception
    i+=1
print(x_test.shape)

x_test -= train_mean
y_test_0 = model.predict(x_test,batch_size=64)
y_pred_0 = np.array(y_test_0>np.array(thresh_2_val))

x_test = []
x_test = np.zeros((30000,224,224,3), np.float32)


i = 0 
for f, tags in tqdm(df_test.values[:30000], miniters=1000):
    img = cv2.imread('test-jpg-1/{}.jpg'.format(f))
    x_test[i,:,:,:] = np.array(cv2.resize(img, (224, 224)),np.float32)/255.#139 minimum size for inception
    i+=1
print(x_test.shape)

x_test -= train_mean
y_test_1 = model.predict(x_test,batch_size=64)
y_pred_1 = np.array(y_test_1>np.array(thresh_2_val))

x_test = []
x_test = np.zeros((30000,224,224,3), np.float32)


i = 0 
for f, tags in tqdm(df_test.values[:30000], miniters=1000):
    img = cv2.imread('test-jpg-2/{}.jpg'.format(f))
    x_test[i,:,:,:] = np.array(cv2.resize(img, (224, 224)),np.float32)/255.#139 minimum size for inception
    i+=1
print(x_test.shape)

x_test -= train_mean
y_test_2 = model.predict(x_test,batch_size=64)
y_pred_2 = np.array(y_test_2>np.array(thresh_2_val))

x_test = []
x_test = np.zeros((30000,224,224,3), np.float32)


i = 0 
for f, tags in tqdm(df_test.values[:30000], miniters=1000):
    img = cv2.imread('test-jpg-3/{}.jpg'.format(f))
    x_test[i,:,:,:] = np.array(cv2.resize(img, (224, 224)),np.float32)/255.#139 minimum size for inception
    i+=1
print(x_test.shape)

x_test -= train_mean
y_test_3 = model.predict(x_test,batch_size=64)
y_pred_3 = np.array(y_test_3>np.array(thresh_2_val))

x_test = []
x_test = np.zeros((30000,224,224,3), np.float32)


i = 0 
for f, tags in tqdm(df_test.values[:30000], miniters=1000):
    img = cv2.imread('test-jpg-4/{}.jpg'.format(f))
    x_test[i,:,:,:] = np.array(cv2.resize(img, (224, 224)),np.float32)/255.#139 minimum size for inception
    i+=1
print(x_test.shape)

x_test -= train_mean
y_test_4 = model.predict(x_test,batch_size=64)
y_pred_4 = np.array(y_test_4>np.array(thresh_2_val))

y_pred_0= np.array(y_pred_0,np.uint)
y_pred_1= np.array(y_pred_1,np.uint)
y_pred_2= np.array(y_pred_2,np.uint)
y_pred_3= np.array(y_pred_3,np.uint)
y_pred_4= np.array(y_pred_4,np.uint)

x_test = []
x_test = np.zeros((31191,224,224,3), np.float32)


i = 0 
for f, tags in tqdm(df_test.values[30000:], miniters=1000):
    img = cv2.imread('test-jpg/{}.jpg'.format(f))
    x_test[i,:,:,:] = np.array(cv2.resize(img, (224, 224)),np.float32)/255.#139 minimum size for inception
    i+=1
print(x_test.shape)

x_test -= train_mean
y_test_0_1 = model.predict(x_test,batch_size=64)
y_pred_0_1 = np.array(y_test_0_1>np.array(thresh_2_val))

x_test = []
x_test = np.zeros((31191,224,224,3), np.float32)


i = 0 
for f, tags in tqdm(df_test.values[30000:], miniters=1000):
    img = cv2.imread('test-jpg-1/{}.jpg'.format(f))
    x_test[i,:,:,:] = np.array(cv2.resize(img, (224, 224)),np.float32)/255.#139 minimum size for inception
    i+=1
print(x_test.shape)

x_test -= train_mean
y_test_1_1 = model.predict(x_test,batch_size=64)
y_pred_1_1 = np.array(y_test_1_1>np.array(thresh_2_val))

x_test = []
x_test = np.zeros((31191,224,224,3), np.float32)


i = 0 
for f, tags in tqdm(df_test.values[30000:], miniters=1000):
    img = cv2.imread('test-jpg-2/{}.jpg'.format(f))
    x_test[i,:,:,:] = np.array(cv2.resize(img, (224, 224)),np.float32)/255.#139 minimum size for inception
    i+=1
print(x_test.shape)

x_test -= train_mean
y_test_2_1 = model.predict(x_test,batch_size=64)
y_pred_2_1 = np.array(y_test_2_1>np.array(thresh_2_val))

x_test = []
x_test = np.zeros((31191,224,224,3), np.float32)


i = 0 
for f, tags in tqdm(df_test.values[30000:], miniters=1000):
    img = cv2.imread('test-jpg-3/{}.jpg'.format(f))
    x_test[i,:,:,:] = np.array(cv2.resize(img, (224, 224)),np.float32)/255.#139 minimum size for inception
    i+=1
print(x_test.shape)

x_test -= train_mean
y_test_3_1 = model.predict(x_test,batch_size=64)
y_pred_3_1 = np.array(y_test_3_1>np.array(thresh_2_val))

x_test = []
x_test = np.zeros((31191,224,224,3), np.float32)


i = 0 
for f, tags in tqdm(df_test.values[30000:], miniters=1000):
    img = cv2.imread('test-jpg-4/{}.jpg'.format(f))
    x_test[i,:,:,:] = np.array(cv2.resize(img, (224, 224)),np.float32)/255.#139 minimum size for inception
    i+=1
print(x_test.shape)

x_test -= train_mean
y_test_4_1 = model.predict(x_test,batch_size=64)
y_pred_4_1 = np.array(y_test_4_1>np.array(thresh_2_val))

y_pred_0_1= np.array(y_pred_0_1,np.uint)
y_pred_1_1= np.array(y_pred_1_1,np.uint)
y_pred_2_1= np.array(y_pred_2_1,np.uint)
y_pred_3_1= np.array(y_pred_3_1,np.uint)
y_pred_4_1= np.array(y_pred_4_1,np.uint)

y_pred_0 = np.concatenate([y_pred_0,y_pred_0_1])
y_pred_1 = np.concatenate([y_pred_1,y_pred_1_1])
y_pred_2 = np.concatenate([y_pred_2,y_pred_2_1])
y_pred_3 = np.concatenate([y_pred_3,y_pred_3_1])
y_pred_4 = np.concatenate([y_pred_4,y_pred_4_1])


y_sum = y_pred_0 + y_pred_1 + y_pred_2 + y_pred_3 + y_pred_4

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

y_pred = (y_sum>=3)

labels_np = np.array(labels)
preds = [' '.join(labels_np[np.array(y_pred[i,:],bool)]) for i in range(y_pred.shape[0])]
subm = pd.DataFrame()
subm['image_name'] = df_test.values[:,0]
subm['tags'] = preds
subm.to_csv('submission_xception_224_1.csv', index=False)
#test set score:0.92646

y_pred = y_pred_0

labels_np = np.array(labels)
preds = [' '.join(labels_np[np.array(y_pred[i,:],bool)]) for i in range(y_pred.shape[0])]
subm = pd.DataFrame()
subm['image_name'] = df_test.values[:,0]
subm['tags'] = preds
subm.to_csv('submission_xception_224_2.csv', index=False)
#test set score:0.92516
