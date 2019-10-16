# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:20:10 2019

@author: Brolof
"""

import h5py
import os
import csv
import torchvision.transforms as transforms
import numpy
from scipy.io import wavfile
from Scripts.front_end_processing import logfbank
from Scripts.front_end_processing import normalizeSpectrogram
from Scripts.front_end_processing import padSpectrogram
from Scripts.import_data import ImportData

dataset_path = "./data/"
#output_path = "./input_data/"

transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.0],
                 std=[0.5])])

subFolderList = []
for x in os.listdir(dataset_path):
    if os.path.isdir(dataset_path + '/' + x):
        subFolderList.append(x)   
    
    
total = 0
data_ID = []
label_index = 0
label_index_ID_table = {}
index_label_ID_table = {}
for x in (subFolderList[0:35]): #(subFolderList[0:n_words])
    # get all the wave filenames
    all_files = [x + '/' + y for y in os.listdir(dataset_path + x) if '.wav' in y]
    total += len(all_files)
    
    data_ID += all_files
    label_index_ID_table[label_index] = x
    index_label_ID_table[x] = label_index
    label_index += 1

nfft = 512
nfilt = 70
window_size = 0.02 # s
step_size = 0.01 # s
expectedRows = nfilt
expectedCols = 99    

n_samples = len(data_ID)

print('Starting transformation process:')

hdf5_dir = "./input_data_hdf5_70m/"
with h5py.File(hdf5_dir + "fileArray.hdf5", "w") as file:
    dsetData = file.create_dataset('data', (n_samples,1,70,99))
    dsetLabel = file.create_dataset('label', (n_samples,), dtype = 'i8')
    for i, ID in enumerate(data_ID[0:n_samples],0):
        samplerate, test_sound  = wavfile.read('./data/' + ID)
        #print(ID)
        spec = logfbank(test_sound,samplerate,
                    winlen = window_size, 
                    winstep = step_size,
                    nfilt = nfilt, 
                    nfft = nfft)
        spec = normalizeSpectrogram(spec.T)
        spec = padSpectrogram(spec,expectedRows, expectedCols)
        file_name = ID.replace('.wav','')
        tensor = transform(spec).float().long()
        
        #torch.save(tensor, output_path + file_name + ".pt") 
        dsetData[i,:,:,:] = tensor
        dsetLabel[i] = index_label_ID_table[ID.split("/")[0]]
        
        if (i+1) % (n_samples/20) == 0:
            print("{0:.3f}% completed".format(i/n_samples*100.0))
    #numpy.savetxt(output_path + file_name + ".png", spec, delimiter=",")
    
    print("100.000% completed")
    print("Finished processing data")
    


    
with open(hdf5_dir + "index_label_ID_table" + ".csv", "w") as outfile:
    writer = csv.writer(outfile)
    for key, val in index_label_ID_table.items():
        writer.writerow([key, val])

with open(hdf5_dir + "label_index_ID_table" + ".csv", "w") as outfile:
    writer = csv.writer(outfile)
    for key, val in label_index_ID_table.items():
        writer.writerow([key, val])
        
#%%
import h5py
import torch
hdf5_dir = "./input_data_hdf5_70m/"
with h5py.File(hdf5_dir + "filePT.hdf5", "r") as file:
    print(file.items())
    x = torch.tensor(file['data'][2,:,:,:])
    print(x)
    z = file['data'][2,:,:,:]
    print(type(z))
    y = file['label'][2]
#one_spec = numpy.array(file[file_name])
#print(type(one_spec))
#%%
file = h5py.File(hdf5_dir + "filePT.hdf5", "w")
dset = file.create_dataset('data', (10,1,70,99))
file.close()


#%%

with h5py.File(hdf5_dir + "filePT.hdf5", "w") as file:
    for i, ID in enumerate(data_ID[0:n_samples],0):
        samplerate, test_sound  = wavfile.read('./data/' + ID)
        #print(ID)
        spec = logfbank(test_sound,samplerate,
                    winlen = window_size, 
                    winstep = step_size,
                    nfilt = nfilt, 
                    nfft = nfft)
        spec = normalizeSpectrogram(spec.T)
        spec = padSpectrogram(spec,expectedRows, expectedCols)
        file_name = ID.replace('.wav','')
        tensor = transform(spec).float().long()
        file.create_dataset(file_name,tensor)
        #torch.save(tensor, output_path + file_name + ".pt") 
        
        if (i+1) % n_samples/20 == 0:
            print("{0:.3f}% completed".format(i/total*100.0))
    #numpy.savetxt(output_path + file_name + ".png", spec, delimiter=",")
    
    print("Finished processing data")
    
    
#%%
    
for i in range(0,1000):
    if (i) % (1000/20) == 0:
            print("{0:.3f}% completed".format(i/1000*100.0))
    
#%%
dataset_path = "./data/"
VERBOSE = False      
_,partition, _, _ = ImportData.importData(dataset_path,35)# IDs            
    
train_idx = partition['train']
validation_idx = partition['validation']
test_idx = partition['test']

with open(hdf5_dir + "train_idx" + ".csv", "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(train_idx)
    
with open(hdf5_dir + "validation_idx" + ".csv", "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(validation_idx)

with open(hdf5_dir + "test_idx" + ".csv", "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(test_idx)
    
#%%
    
path_to_csv =  "./input_data_hdf5_70m/" + "test_idx.csv"
with open(path_to_csv, newline='') as csvfile:
    data = list(csv.reader(csvfile))
    data = numpy.asarray(data)
    #data = list(map(int, data))

#%%