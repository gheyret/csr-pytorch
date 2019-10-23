# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:33:32 2019

@author: Brolof
"""

import torch
import time
import matplotlib.pyplot as plt
from data.front_end_processing import fbank
from data.front_end_processing import logfbank
from data.front_end_processing import logfbankGPU
from data.front_end_processing import createSpectrogram
from data.front_end_processing import normalizeSpectrogram
from data.front_end_processing import padSpectrogram
from scipy.io import wavfile
nfilt = 70
nfft = 512
window_size = 20
step_size = 10

expectedRows = nfilt
expectedCols = 99

path1 = './data/' + "four/4e02d62d_nohash_0.wav"
path2 = './data/' + "happy/0bd689d7_nohash_0.wav"
samplerate, test_sound1  = wavfile.read(path1)
samplerate, test_sound2  = wavfile.read(path2)
#%% Spectrogram using numpy's signal.spectrogram

spec = createSpectrogram(path1, window_size, step_size, nfft)
spec = normalizeSpectrogram(spec.T)
spec = padSpectrogram(spec,nfft/2+1, expectedCols)
plt.figure(1)
plt.clf()
plt.imshow(spec, cmap=plt.cm.jet, aspect='auto', origin='lower')
plt.axis('off')

spec = createSpectrogram(path2, window_size, step_size, nfft)
spec = normalizeSpectrogram(spec.T)
spec = padSpectrogram(spec,nfft/2+1, expectedCols)
plt.figure(2)
plt.clf()
plt.imshow(spec, cmap=plt.cm.jet, aspect='auto', origin='lower')
plt.axis('off')

#%% Log filterbanks

spec = logfbank(test_sound1,samplerate,
                winlen = window_size/1000, 
                winstep = step_size/1000,
                nfilt = nfilt, 
                nfft = nfft)
spec = normalizeSpectrogram(spec.T)
spec = padSpectrogram(spec,expectedRows, expectedCols)
plt.figure(3)
plt.clf()
plt.imshow(spec, cmap=plt.cm.jet, aspect='auto', origin='lower')
plt.axis('off')

spec = logfbank(test_sound2,samplerate,
                winlen = window_size/1000, 
                winstep = step_size/1000,
                nfilt = nfilt, 
                nfft = nfft)
spec = normalizeSpectrogram(spec.T)
spec = padSpectrogram(spec,expectedRows, expectedCols)
plt.figure(4)
plt.clf()
plt.imshow(spec, cmap=plt.cm.jet, aspect='auto', origin='lower')
plt.axis('off')

#%% filter bank

spec, _ = fbank(test_sound1,samplerate,
                winlen = window_size/1000, 
                winstep = step_size/1000,
                nfilt = nfilt, 
                nfft = nfft)
spec = normalizeSpectrogram(spec.T)
spec = padSpectrogram(spec,expectedRows, expectedCols)
plt.figure(5)
plt.clf()
plt.imshow(spec, cmap=plt.cm.jet, aspect='auto', origin='lower')
plt.axis('off')

spec, _ = fbank(test_sound2,samplerate,
                winlen = window_size/1000, 
                winstep = step_size/1000,
                nfilt = nfilt, 
                nfft = nfft)
spec = normalizeSpectrogram(spec.T)
spec = padSpectrogram(spec,expectedRows, expectedCols)
plt.figure(6)
plt.clf()
plt.imshow(spec, cmap=plt.cm.jet, aspect='auto', origin='lower')
plt.axis('off')

#%% Testing the speed of logfbank
# waveform is in test_sound2

test_tensor = torch.FloatTensor(test_sound2).cuda()
print(test_tensor.size())


spec = logfbankGPU(test_tensor,samplerate,
                winlen = window_size/1000, 
                winstep = step_size/1000,
                nfilt = nfilt, 
                nfft = nfft)


#%%
start_time = time.time()
for i in range(0,100):
    spec = logfbank(test_sound2,samplerate,
                    winlen = window_size/1000, 
                    winstep = step_size/1000,
                    nfilt = nfilt, 
                    nfft = nfft)
end_time = time.time()-start_time
print(end_time*1000/100)