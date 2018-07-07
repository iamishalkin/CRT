# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 14:59:32 2018

@author: ivan.mishalkin
"""

#https://datascience.stackexchange.com/questions/26366/training-an-rnn-with-examples-of-different-lengths-in-keras

import os
import numpy as np
import pandas as pd
from random import shuffle
import random
random.seed(42)
import librosa
meta = pd.read_csv('C:/CRT/data_v_7_stc/meta/meta.txt', sep='\t', header=None)

directory = 'C:/CRT/data_v_7_stc/audio/'

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

list_of_fn = meta.iloc[:,0].tolist()
shuffle(list_of_fn)
from sklearn.cross_validation import train_test_split
x_train ,x_test = train_test_split(list_of_fn,test_size=0.2)

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
lb = LabelEncoder()
y = np_utils.to_categorical(lb.fit_transform(meta.iloc[:,4]))

def train_generator(directory, list_of_fn, y, bands = 60):
    window_size = 512
    
    for fn in list_of_fn:
        mfccs = []
        path = directory + fn
        sound_clip,s = librosa.load(path)
        for (start,end) in windows(sound_clip,window_size):
            end = int(end)
            start = int(start)
            if(len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = bands).T.flatten()[:, np.newaxis].T
                mfccs.append(mfcc)
    
        features = np.asarray(mfccs).reshape(1, len(mfccs),bands*2)
        #print(len(mfccs))
        y_resh = np.asarray(y[meta.iloc[:,0].tolist().index(fn)]).reshape(1, 8)
        yield features, y_resh




from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint 
from sklearn import metrics
from keras.optimizers import Adam

model = Sequential()

model.add(LSTM(32, return_sequences=True, input_shape=(None, 120)))
model.add(LSTM(16))
model.add(Dense(8, activation='softmax'))

print(model.summary(90))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

saveBestModel = ModelCheckpoint("best.kerasModelWeights",
                                monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)

model.fit_generator(train_generator(directory, x_train, y, bands = 60), 
                    validation_data=train_generator(directory, x_test, y, bands = 60),
                    callbacks=[saveBestModel],
                    steps_per_epoch=10, epochs=100, verbose=1, validation_steps=20)


#tr_features = extract_features(directory, list_of_fn[0:3])
test_directory = 'C:/CRT/data_v_7_stc/test/'
filenames = [i.decode("utf-8") for i in os.listdir(os.fsencode(test_directory))]
#test = extract_features(test_directory, list_of_fn)

def test_generator(directory, list_of_fn, bands = 60):
    window_size = 512
    
    for fn in list_of_fn:
        mfccs = []
        path = directory + fn
        sound_clip,s = librosa.load(path)
        for (start,end) in windows(sound_clip,window_size):
            end = int(end)
            start = int(start)
            if(len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = bands).T.flatten()[:, np.newaxis].T
                mfccs.append(mfcc)
    
        features = np.asarray(mfccs).reshape(1, len(mfccs),bands*2)
        #print(len(mfccs))
        
        yield features
        
predict = model.predict_generator(test_generator(test_directory, filenames, bands = 60), steps=len(filenames))