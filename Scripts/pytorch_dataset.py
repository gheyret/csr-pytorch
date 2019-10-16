# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:23:58 2019

@author: Brolof
"""

from torch.utils import data
import torchvision.transforms as transforms
import time
import torch


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transformData = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean=[0.0],
                         std=[0.5])])
        self.transformLabels = transforms.ToTensor()
        self.nfft = 512
        self.nfilt = 70
        self.window_size = 0.02 # s
        self.step_size = 0.01 # s
        self.expectedRows = self.nfilt
        self.expectedCols = 99
    
    
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label       
        # create spectogram
        start_time = time.time()
        #samplerate, test_sound  = wavfile.read('./data/' + ID + ".wav")
        #spec = logfbank(test_sound,samplerate,
        #        winlen = self.window_size, 
        #        winstep = self.step_size,
        #        nfilt = self.nfilt, 
        #        nfft = self.nfft)
        #spec = normalizeSpectrogram(spec.T)
        #spec = padSpectrogram(spec,self.expectedRows, self.expectedCols)
        #x = self.transformData(spec).float().cuda()
        
        x = torch.load('./input_data/' + ID + ".pt").cpu()
        #print("is cuda? :", x.is_cuda)
        #print("Spectrogram: ",(time.time()-start_time)*1000, " ms")
        #print(x.size())
        y = self.labels[ID]
        return x, y
    
  