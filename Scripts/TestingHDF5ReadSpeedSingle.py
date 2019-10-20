
from Scripts.import_data import ImportData
import h5py
import random
import torch
import time

def load_data_set_path(dataset_path):
    # VERBOSE = False
    partition, labels, label_index_ID_table = ImportData.importData(dataset_path,35)# IDs
    # labels = # Labels
    return partition, labels


def load_data_set_indexes(dataset_path):
    partition = {'train': ImportData.csvToList(dataset_path + "train_idx.csv"),
                 'validation': ImportData.csvToList(dataset_path + "validation_idx.csv"),
                 'test': ImportData.csvToList(dataset_path + "test_idx.csv")}

    return partition

test_ID = []
with open("../data/" + "testing_list.txt", "r") as f:
    if f.mode == 'r':
        contents = f.readlines()

    print("number of test elements: %d" % len(contents))

    for s in contents:
        s = s.replace("\n", '')
        s = s.replace(".wav", '')
        test_ID.extend([s])

batch_size = 400
dataset_path = "../data/"
datasethdf5_path = "../input_data_hdf5_70m/"

partition_path, labels = load_data_set_path(dataset_path)

partition_idx = load_data_set_indexes(datasethdf5_path)

#############################

n_samples = 10000
filesToReadIdx = random.sample(range(1, n_samples), batch_size)
print('Storing 10000 samples wav in one large ndarray. But format: [samples,16000]')
list_ID = partition_path['train']
#print(len(list_ID))
start_time = time.time()
with h5py.File(datasethdf5_path + "testWav12.hdf5", "r") as file:
    for i in filesToReadIdx:
        x = file['ds/data'][i,:]
print("End time: ", (time.time() - start_time))

n_samples = 10000
filesToReadIdx = random.sample(range(1, n_samples), batch_size)
print('Storing 10000 samples wav in one large ndarray. But format: [samples,16000] and chunked 1,16000')
list_ID = partition_path['train']
#print(len(list_ID))
start_time = time.time()
with h5py.File(datasethdf5_path + "testChunking.hdf5", "r") as file:
    for i in filesToReadIdx:
        x = file['ds/data'][i,:]
print("End time: ", (time.time() - start_time))


n_samples = 10000
filesToReadIdx = random.sample(range(1, n_samples), batch_size)
print('Storing 10000 samples wav in one large ndarray. But format: [samples,16000] and chunked 5,16000')
list_ID = partition_path['train']
#print(len(list_ID))
start_time = time.time()
with h5py.File(datasethdf5_path + "testChunking2.hdf5", "r") as file:
    for i in filesToReadIdx:
        x = file['ds/data'][i,:]
print("End time: ", (time.time() - start_time))

n_samples = 10000
filesToReadIdx = random.sample(range(1, n_samples), batch_size)
print('Storing 10000 samples wav in one large ndarray. But format: [samples,16000] and chunked 100,16000')
list_ID = partition_path['train']
#print(len(list_ID))
start_time = time.time()
with h5py.File(datasethdf5_path + "testChunking3.hdf5", "r") as file:
    for i in filesToReadIdx:
        x = file['ds/data'][i,:]
print("End time: ", (time.time() - start_time))

n_samples = 10000
filesToReadIdx = random.sample(range(1, n_samples), batch_size)
print('Storing 10000 samples wav in one large ndarray. But format: [samples,16000] and chunked 1000,16000')
list_ID = partition_path['train']
#print(len(list_ID))
start_time = time.time()
with h5py.File(datasethdf5_path + "testChunking4.hdf5", "r") as file:
    for i in filesToReadIdx:
        x = file['ds/data'][i,:]
print("End time: ", (time.time() - start_time))

n_samples = 10000
filesToReadIdx = random.sample(range(1, n_samples), batch_size)
print('Storing 10000 samples wav in one large ndarray. But format: [samples,16000] and chunked 5000,16000')
list_ID = partition_path['train']
#print(len(list_ID))
start_time = time.time()
with h5py.File(datasethdf5_path + "testChunking5.hdf5", "r") as file:
    for i in filesToReadIdx:
        x = file['ds/data'][i,:]
print("End time: ", (time.time() - start_time))

n_samples = 10000
filesToReadIdx = random.sample(range(1, n_samples), batch_size)
print('Storing 10000 samples wav in one large ndarray. But format: [samples,16000] and chunked 10000,16000')
list_ID = partition_path['train']
#print(len(list_ID))
start_time = time.time()
with h5py.File(datasethdf5_path + "testChunking6.hdf5", "r") as file:
    for i in filesToReadIdx:
        x = file['ds/data'][i,:]
print("End time: ", (time.time() - start_time))
print('Finished')
