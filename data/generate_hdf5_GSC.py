"""
Created on Tue Sep 24 11:20:10 2019

Using GoogleSpeechCommands
Used to generate HDF5 file.
Each sample in the hdf5 file is a spectrogram with mel-scale filters in one dim and frames in the other

"""

import h5py
import os
import csv
from data.import_data_GSC import import_data
from scipy.io import wavfile
import numpy
from os.path import dirname


dataset_path_relative = "/data/GoogleSpeechCommands/wav_format/"
hdf5_path_relative = "/data/GoogleSpeechCommands/hdf5_format/"

parent_dir = dirname(os.getcwd())
dataset_path = parent_dir + dataset_path_relative
hdf5_dir = parent_dir + hdf5_path_relative
if not os.path.exists(dataset_path):
    grandparent_dir = dirname(dirname(os.getcwd()))
    dataset_path = grandparent_dir + dataset_path_relative
    hdf5_dir = grandparent_dir + hdf5_path_relative

# Generate a list of all subfolders
subFolderList = []
for x in os.listdir(dataset_path):
    if os.path.isdir(dataset_path + '/' + x):
        subFolderList.append(x)

# Generate a list of the ID of all samples
# Also generate label_index dictionary and reverse
total = 0
data_ID = []
label_index = 0
label_index_ID_table = {}
index_label_ID_table = {}
for x in (subFolderList[0:35]):  # (subFolderList[0:n_words])
    # get all the wave filenames
    all_files = [x + '/' + y for y in os.listdir(dataset_path + x) if '.wav' in y]
    total += len(all_files)
    all_files = [z.replace(".wav", '') for z in all_files]  # remove .wav
    data_ID += all_files
    label_index_ID_table[label_index] = x
    index_label_ID_table[x] = label_index
    label_index += 1

n_samples = len(data_ID)

# Generate the HDF5 file.
output_file_name = "allWavIdx.hdf5"
if not os.path.exists(hdf5_dir + output_file_name):
    # Generating a hdf5 file with a single ndarray [index,1,70,99] in group 'data'
    print('Starting transformation process:')
    with h5py.File(hdf5_dir + output_file_name, "w") as file:
        dsetData = file.create_dataset('ds/data', (n_samples, 16000))
        dsetLabel = file.create_dataset('ds/label', (n_samples,), dtype = 'i8')
        for i, ID in enumerate(data_ID[0:n_samples], 0):
            samplerate, test_sound = wavfile.read(dataset_path + ID + ".wav")
            z = numpy.zeros((1, 16000))
            z[0, :test_sound.shape[0]]=test_sound
            file_name = ID.replace('.wav', '')
            dsetData[i, :] = z
            dsetLabel[i] = index_label_ID_table[ID.split("/")[0]]

            if (i + 1) % (n_samples / 100) == 0:
                print("{0:.3f}% completed".format(i / n_samples * 100.0))

        print("100.000% completed")
        print("Finished processing data")


# Generate some csv files:
if not os.path.exists(hdf5_dir + "index_label_ID_table"):
    with open(hdf5_dir + "index_label_ID_table" + ".csv", "w", newline='') as outfile:
        writer = csv.writer(outfile)
        for key, val in index_label_ID_table.items():
            writer.writerow([key, val])

if not os.path.exists(hdf5_dir + "label_index_ID_table"):
    with open(hdf5_dir + "label_index_ID_table" + ".csv", "w", newline='') as outfile:
        writer = csv.writer(outfile)
        for key, val in label_index_ID_table.items():
            writer.writerow([key, val])

partition, labels, _ = import_data(dataset_path, return_index=True)  # IDs
train_idx = partition['train']
validation_idx = partition['validation']
test_idx = partition['test']
if not os.path.exists(hdf5_dir + "train_idx"):
    with open(hdf5_dir + "train_idx" + ".csv", "w", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(train_idx)

if not os.path.exists(hdf5_dir + "validation_idx"):
    with open(hdf5_dir + "validation_idx" + ".csv", "w", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(validation_idx)

if not os.path.exists(hdf5_dir + "test_idx"):
    with open(hdf5_dir + "test_idx" + ".csv", "w", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(test_idx)