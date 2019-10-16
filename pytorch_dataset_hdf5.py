# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:17:39 2019

@author: Brolof
"""
import h5py
import torch
from torch.utils import data


class Dataset(data.Dataset):
    
    def __init__(self, hdf5file_path, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs  
        self.hdf5file_path = hdf5file_path
        self.dataset = None

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        #TODO: Will all workers be able to work with the file, or are they waiting for each other?
        #print(torch.utils.data.get_worker_info())
        if self.dataset is None:
            self.dataset = h5py.File(self.hdf5file_path, "r")

        ID = self.list_IDs[index]

        x = torch.tensor(self.dataset['data'][ID, :, :, :]).cpu()
        
        y = self.dataset['label'][ID]

        return x,y