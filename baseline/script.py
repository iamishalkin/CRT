# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 08:45:40 2018

@author: ivan.mishalkin
"""

import os
import numpy as np
import pandas as pd
import re
import librosa
meta = pd.read_csv('C:/CRT/data_v_7_stc/meta/meta.txt', sep='\t', header=None)

directory = 'C:/CRT/data_v_7_stc/audio/'

def mfcc(filename, directory):
    path = directory + filename
    X, sample_rate = librosa.load(path, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    feature = mfccs
    return feature

X_train = []
y_train = []

for i in range(0,len(meta)):
    filename = meta.iloc[i, 0]
    y_train.append(meta.iloc[i, 4])
    X_train.append(mfcc(filename, directory))
X_train=pd.DataFrame(X_train)
y_train=pd.DataFrame(y_train)

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint 
from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto( inter_op_parallelism_threads=2, intra_op_parallelism_threads=2)))

lb = LabelEncoder()

y = np_utils.to_categorical(lb.fit_transform(y_train.iloc[:,0]))

X_tr, val_x, y_tr, val_y = train_test_split(X_train, y, test_size=0.2)

num_labels = y.shape[1]
filter_size = 2

# build model
model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

saveBestModel = ModelCheckpoint("best.kerasModelWeights", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)



model.fit(X_tr, y_tr, batch_size=64,
                callbacks=[saveBestModel],
                epochs=500,
                validation_data=(val_x, val_y))
    

# Prediction part

test_directory = os.fsencode('C:/CRT/data_v_7_stc/test/')

test = []
for file in os.listdir(test_directory):
    filename = os.fsdecode(file)
    test.append(mfcc(filename, 'C:/CRT/data_v_7_stc/test/'))

test = pd.DataFrame(test)


filenames = [i.decode("utf-8") for i in os.listdir(test_directory)]
pred = model.predict_classes(test)
pred_proba = model.predict_proba(test).max(axis=1)
probs = list(pred_proba)
labels = list(lb.inverse_transform(pred))

final = pd.DataFrame(list(zip(filenames, probs, labels)),columns=['lst1Title', 'lst2Title', 'lst3Title'])

final.to_csv('C:/CRT/result.txt', sep='\t', header=False, index=False)















