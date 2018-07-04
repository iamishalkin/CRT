# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 18:24:16 2018

@author: ivan.mishalkin
"""
# http://aqibsaeed.github.io/2016-09-24-urban-sound-classification-part-2/
import os
import numpy as np
import pandas as pd
import re
import librosa
meta = pd.read_csv('C:/CRT/data_v_7_stc/meta/meta.txt', sep='\t', header=None)

directory = 'C:/CRT/data_v_7_stc/audio/'

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

import glob

list_of_fn = meta.iloc[:,0]
def extract_features(directory, list_of_fn, file_ext="*.wav", bands = 60, frames = 41, meta):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    
    for fn in list_of_fn:
        path = directory + fn
        sound_clip,s = librosa.load(path)
        label = meta.loc[meta.iloc[:,0]==fn, 4]
        for (start,end) in windows(sound_clip,window_size):
            if(len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                logspec = librosa.logamplitude(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)
                labels.append(label)
            
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    
    return np.array(features), np.array(labels)



















bands = 60
frames = 41
window_size = 512 * (frames - 1)
log_specgrams = []
labels = []

def extract_features(filename, directory):
#    path = directory + filename
#    X, sample_rate = librosa.load(path, res_type='kaiser_fast')
#    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
#    feature = mfccs
#    return feature
    path = directory + filename
    sound_clip,s = librosa.load(path)
    for (start,end) in windows(sound_clip,window_size):
        if(len(sound_clip[start:end]) == window_size):
            signal = sound_clip[start:end]
            melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
            logspec = librosa.logamplitude(melspec)
            logspec = logspec.T.flatten()[:, np.newaxis].T
            log_specgrams.append(logspec)
    

X_train = []
y_train = []

for i in range(0,len(meta)):
    filename = meta.iloc[i, 0]
    y_train.append(meta.iloc[i, 4])
    X_train.append(mfcc(filename, directory))
X_train=pd.DataFrame(X_train)
y_train=pd.DataFrame(y_train)