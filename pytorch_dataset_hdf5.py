# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:17:39 2019

@author: Brolof
"""
import h5py
import torch
from torch.utils import data

import numpy
import time
import matplotlib.pyplot as plt
from custom_transforms import RangeNormalize

class Dataset(data.Dataset):
    
    def __init__(self, hdf5file_path, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs  
        self.hdf5file_path = hdf5file_path
        self.dataset = None
        self.range_normalize = RangeNormalize(-1, 1)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):

        if self.dataset is None:
            self.dataset = h5py.File(self.hdf5file_path, "r")

        ID = self.list_IDs[index]
        x = torch.tensor(self.dataset['data'][ID, :, :, :]).cpu()
        y = self.dataset['label'][ID]

        #x = self.range_normalize(x)

        if False:
            #spec1 = numpy.asarray(x)  ###########################
            spec2 = numpy.asarray(x)
            plt.figure(1)
            plt.clf()
            plt.subplot(211)
            im_spec = plt.imshow(spec1[0, :, :], cmap=plt.cm.jet, interpolation='none', aspect='auto', origin='lower')
            plt.colorbar(im_spec)
            plt.subplot(212)
            im_spec2 = plt.imshow(spec2[0, :, :], cmap=plt.cm.jet, interpolation='none', aspect='auto', origin='lower')
            plt.colorbar(im_spec2)
            plt.show()
            # x = torch.tensor(self.dataset['data'][ID, :, :, :]).cpu()

        return x,y