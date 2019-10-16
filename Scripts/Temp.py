# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:47:08 2019

@author: Brolof
"""

import matplotlib.pyplot as plt
from matplotlib.backend_bases import RendererBase
from scipy import signal
from scipy.io import wavfile
#import soundfile as sf
import os
import numpy
from PIL import Image
from scipy.fftpack import fft

audio_path = "../Dataset/Google Speech Commands/data_speech_commands_v0.02/"
train_path = "./input/train/"
test_path = "./input/test/"

# List the name of all subfolders
subFolderList = []
for x in os.listdir(audio_path):
    if os.path.isdir(audio_path + '/' + x):
        subFolderList.append(x)

x = os.path.abspath(train_path)
y = os.path.abspath(audio_path)

# Prepare train and test folders
if not os.path.exists(train_path):
    os.makedirs(train_path)

if not os.path.exists(test_path):
    os.makedirs(test_path)
    
n_words = 15
n_samples_train = 10
n_samples_test = 5


# Pull sample audio from each word
print(subFolderList[0:n_words])
sample_audio = []
total = 0
for x in subFolderList[0:n_words]:
    # get all the wave files
    all_files = [y for y in os.listdir(audio_path + x) if '.wav' in y]
    total += len(all_files)
    # collect the first file from each dir
    sample_audio.append(audio_path  + x + '/'+ all_files[0])
    
    # show file counts
    print('count: %d : %s' % (len(all_files), x ))
print(total)


# Function for computing the log filterbanks
def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, numpy.log(spec.T.astype(numpy.float32) + eps)

# Look at the top 9 different words in spectrogram format
fig = plt.figure(figsize=(10,10))

# for each of the samples
for i, filepath in enumerate(sample_audio[:9]):
    # Make subplots
    plt.subplot(3,3,i+1)
    
    # pull the labels
    label = filepath.split('/')[-2]
    plt.title(label)
    
    # create spectogram
    samplerate, test_sound  = wavfile.read(filepath)
    _, spectrogram = log_specgram(test_sound, samplerate)
    
    plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    plt.axis('off')


