# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 18:24:16 2018

@author: ivan.mishalkin
"""
# http://aqibsaeed.github.io/2016-09-24-urban-sound-classification-part-2/
import os
import numpy as np
import pandas as pd
#import re
import librosa
from tqdm import tqdm
meta = pd.read_csv('C:/CRT/data_v_7_stc/meta/meta.txt', sep='\t', header=None)

directory = 'C:/CRT/data_v_7_stc/audio/'

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

list_of_fn = meta.iloc[:,0].tolist()
def extract_features(directory, list_of_fn, bands = 60, frames = 41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    
    for fn in tqdm(list_of_fn, desc="extract features"):
        path = directory + fn
        sound_clip,s = librosa.load(path)
        for (start,end) in windows(sound_clip,window_size):
            start = int(start)
            end = int(end)
            if(len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                logspec = librosa.amplitude_to_db(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)
            
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in tqdm(range(len(features))):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    
    return np.array(features)

tr_features = extract_features(directory, list_of_fn[0:1])
test_directory = 'C:/CRT/data_v_7_stc/test/'
filenames = [i.decode("utf-8") for i in os.listdir(os.fsencode(test_directory))]
test = extract_features(test_directory, list_of_fn)
#tr_labels = one_hot_encode(tr_labels)

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
from keras.utils import np_utils
y = np_utils.to_categorical(lb.fit_transform(meta.iloc[:,4]))
from sklearn.model_selection import train_test_split
X_tr, val_x, y_tr, val_y = train_test_split(tr_features, y, test_size=0.2)

X_norm = (X_tr - np.mean(X_tr)) / np.std(X_tr)
X_val_norm = (val_x - np.mean(val_x)) / np.std(val_x)
test_norm = (test - np.mean(test)) / np.std(test)




from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras import backend as K
cores = 2
K.set_session(K.tf.Session(config=K.tf.ConfigProto( inter_op_parallelism_threads=cores, intra_op_parallelism_threads=cores))) 

model = Sequential()
model.add(Conv2D(filters=80, kernel_size=(57, 6), strides=(1, 1), padding='valid', input_shape=(60, 41, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 3), strides=(1, 3)))
model.add(Conv2D(filters=80, kernel_size=(1, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
model.add(Flatten())
model.add(Dense(units=5000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=5000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

saveBestModel = ModelCheckpoint("best.kerasModelWeights1", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)


epochs = 2
batch_size = 128

model.fit(X_tr, y_tr,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(val_x, val_y),
          callbacks=[saveBestModel])

#norm fit
model.fit(X_norm, y_tr,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val_norm, val_y),
          callbacks=[saveBestModel])


