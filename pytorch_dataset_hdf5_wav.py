# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:17:39 2019

@author: Brolof
"""
import h5py
import torch
from torch.utils import data
from torchvision.transforms import transforms
from Scripts.front_end_processing import logfbank

import numpy
import time
import matplotlib.pyplot as plt

class Dataset(data.Dataset):

    def __init__(self, hdf5file_path, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs
        self.hdf5file_path = hdf5file_path
        self.transformData = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.0],
                                  std=[0.5])])
        self.dataset = None
        self.nfft = 512
        self.nfilt = 70
        self.window_size = 0.02  # s
        self.step_size = 0.01  # s
        self.expectedRows = self.nfilt
        self.expectedCols = 99
        self.samplerate = 16000

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.hdf5file_path, "r", swmr=True) # SWMR = Single write multiple read

        ID = self.list_IDs[index]
        y = self.dataset['ds/label'][ID]
        test_sound = self.dataset['ds/data'][ID, :]
        spec = logfbank(test_sound, self.samplerate,
                        winlen=self.window_size,
                        winstep=self.step_size,
                        nfilt=self.nfilt,
                        nfft=self.nfft)

        # TODO: Concurrent read access so that several workers can read at the same time from hdf5. SWMR...?
        # TODO: Make values in range [-1,1] or [0,1]
        # TODO: Chunked data, performance gain?
        x = self.transformData(spec.T).float()

        if False:
            spec2 = numpy.asarray(x)
            plt.figure(1)
            plt.clf()
            plt.subplot(211)
            im_spec = plt.imshow(spec2[0,:,:], cmap=plt.cm.jet, interpolation='none', aspect='auto', origin='lower')
            plt.colorbar(im_spec)
            #plt.axis('off')
            plt.subplot(212)
            plt.plot(test_sound)
            #plt.axis('off')
            plt.show()
            time.sleep(0)
            # x = torch.tensor(self.dataset['data'][ID, :, :, :]).cpu()

        return x, y
