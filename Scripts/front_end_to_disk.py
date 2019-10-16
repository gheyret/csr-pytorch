# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:42:39 2019

@author: Brolof
"""
import os
import torch
import torchvision.transforms as transforms
import numpy
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import wavfile
from Scripts.front_end_processing import logfbank
from Scripts.front_end_processing import normalizeSpectrogram
from Scripts.front_end_processing import padSpectrogram

dataset_path = "./data/"
output_path = "./input_data/"

transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.0],
                 std=[0.5])])


if not os.path.exists(output_path):
    os.makedirs(output_path)


subFolderList = []
for x in os.listdir(dataset_path):
    if os.path.isdir(dataset_path + '/' + x):
        subFolderList.append(x)   
    
    
    
total = 0
data_ID = []
for x in (subFolderList[0:35]): #(subFolderList[0:n_words])
    # get all the wave files
    all_files = [x + '/' + y for y in os.listdir(dataset_path + x) if '.wav' in y]
    total += len(all_files)
    
    data_ID += all_files
    
    
    # Make new folder structure for output
    if not os.path.exists(output_path + x):
        os.makedirs(output_path + x)
    
    #label_ID += ([label_index] * len(all_files))
    #label_index_ID_table[label_index] = x
    # show file counts
    #label_index += 1
    #if VERBOSE: print('count: %d : %s' % (len(all_files), x ))
    
# Generate subfolder structure
    # Take each item, place it in the folder as by the first /

nfft = 512
nfilt = 70
window_size = 0.02 # s
step_size = 0.01 # s
expectedRows = nfilt
expectedCols = 99    



for i, ID in enumerate(data_ID[0:1],0):
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

    im1 = Image.fromarray(spec.astype('float64'))
    im1.save(output_path + file_name + ".bmp")
    numpy.savetxt(output_path + file_name + ".csv", spec, delimiter=",")
    tensor = transform(spec).float().cuda()
    torch.save(tensor, output_path + file_name + ".pt") 
    if (i+1) % 1000 == 0:
        print("{0:.3f}% completed".format(i/total*100.0))
#numpy.savetxt(output_path + file_name + ".png", spec, delimiter=",")

print("Finished processing data")

plt.figure(1)
plt.clf()
plt.imshow(spec, cmap=plt.cm.jet, aspect='auto', origin='lower')
plt.axis('off')
    

#%%

import time
import torch
import pandas


ID = "backward/0165e0e8_nohash_0"

start_time = time.time()
tensor_pt = torch.load('./input_data/' + ID + ".pt").cuda()
print("Time .pt: ",(time.time()-start_time)*1000, " ms")

start_time = time.time()
tensor_csv = pandas.read_csv('./input_data/' + ID + ".csv").values
print("Time to load .csv:",(time.time()-start_time)*1000, " ms")
start_time = time.time()
tensor_csv = transform(tensor_csv).cuda()
print("Time to transform .csv to tensor:",(time.time()-start_time)*1000, " ms")

print(tensor_pt.is_cuda)
print(tensor_csv.is_cuda)

#start_time = time.time()
#img = io.imread('./input_data/' + ID + ".png")
#tensor_png = transform(img).cuda()
#print("Time .png:",(time.time()-start_time)*1000, " ms")

