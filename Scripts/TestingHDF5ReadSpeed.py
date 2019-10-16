
from Scripts.import_data import ImportData
from csv_to_list import csvToList
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
    partition = {'train': csvToList(dataset_path + "train_idx.csv"),
                 'validation': csvToList(dataset_path + "validation_idx.csv"),
                 'test': csvToList(dataset_path + "test_idx.csv")}

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

sleep_time = 0
batch_size = 400
dataset_path = "../data/"
datasethdf5_path = "../input_data_hdf5_70m/"

partition_path, labels = load_data_set_path(dataset_path)

partition_idx = load_data_set_indexes(datasethdf5_path)
#############################
filesToReadIdx = random.sample(range(1, 84000), batch_size)
# Storing the data as a huge array
print('Storing 105000 samples in 1 large ndarray')
start_time = time.time()
with h5py.File(datasethdf5_path + "fileArray.hdf5", "r") as file:
    for ID in filesToReadIdx:
        x = torch.tensor(file['data'][ID, :, :, :])

print("End time: ", (time.time() - start_time))
time.sleep(sleep_time)
#############################


list_ID = partition_path['train']
print(len(list_ID))
print('Storing 105000 samples, each sample as its own ndarray, in groups')
start_time = time.time()
with h5py.File(datasethdf5_path + "file.hdf5", "r") as file:
    for i in filesToReadIdx:
        ID = list_ID[i]
        x = torch.tensor(file[ID])
print("End time: ", (time.time() - start_time))
time.sleep(sleep_time)
#############################


filesToReadIdx = random.sample(range(1, 11005), batch_size)
# Storing the data as a huge array
print('Storing 11005 samples in one large ndarray')
start_time = time.time()
with h5py.File(datasethdf5_path + "testIdx.hdf5", "r") as file:
    for ID in filesToReadIdx:
        x = torch.tensor(file['data'][ID, :, :, :])
print("End time: ", (time.time() - start_time))
time.sleep(sleep_time)

#############################

filesToReadIdx = random.sample(range(1, 11005), batch_size)
# Storing the data as a huge array
print('Storing 11005 samples in one large ndarray but now in a subgroup')
start_time = time.time()
with h5py.File(datasethdf5_path + "testIdx2.hdf5", "r") as file:
    for ID in filesToReadIdx:
        x = torch.tensor(file['ds/data'][ID, :, :, :])
print("End time: ", (time.time() - start_time))
time.sleep(sleep_time)


#############################
filesToReadIdx = random.sample(range(1, 11005), batch_size)
list_ID = partition_path['train']
print(len(list_ID))
print('Storing 11005 samples, each sample as its own ndarray, in groups')
start_time = time.time()
with h5py.File(datasethdf5_path + "testPath.hdf5", "r") as file:
    for i in filesToReadIdx:
        ID = test_ID[i]
        x = torch.tensor(file[ID])
print("End time: ", (time.time() - start_time))
time.sleep(sleep_time)

#############################
filesToReadIdx = random.sample(range(1, 11005), batch_size)
print('Storing 10000 samples in one large ndarray. But format: [1,70*samples,99]')
list_ID = partition_path['train']
print(len(list_ID))
start_time = time.time()
with h5py.File(datasethdf5_path + "testGroupIdx.hdf5", "r") as file:
    for i in filesToReadIdx:
        idx_start = i * 70
        idx_end = (i + 1) * 70
        x = torch.tensor(file['data'][:, idx_start:idx_end, :])
print("End time: ", (time.time() - start_time))
time.sleep(sleep_time)

#############################
import math
hdf5fileIdx_path = datasethdf5_path + "testGroupedData.hdf5"
n_samples = 30000
print('Storing 30000 files in groups of 10000 as ndarrays [id,1,70,99]')
filesToReadIdx = random.sample(range(1, n_samples), batch_size)
start_time = time.time()
with h5py.File(datasethdf5_path + "testGroupedData.hdf5", "r") as file:
    for ID in filesToReadIdx:
        dsetIdx = math.ceil(i / (n_samples / 3))
        idx = i % n_samples / 3
        if dsetIdx == 1:
            # torch.save(tensor, output_path + file_name + ".pt")
            x = torch.tensor(file['ds1/data'][idx, :, :, :])
        elif dsetIdx == 2:
            x = torch.tensor(file['ds2/data'][idx, :, :, :])
        elif dsetIdx == 3:
            x = torch.tensor(file['ds3/data'][idx, :, :, :])
        #x = torch.tensor(file['data'][ID, :, :, :])
print("End time: ", (time.time() - start_time))
time.sleep(sleep_time)

#############################
for n in range(1,11):
    print(n)
    time.sleep(sleep_time)
    hdf5_name = "testArraySize" + str(n) + ".hdf5"
    print(hdf5_name)
    n_samples = n*10000
    filesToReadIdx = random.sample(range(1, n_samples), batch_size)
    print('Storing', str(n*10000), ' samples in one large ndarray')
    start_time = time.time()

    file = h5py.File(datasethdf5_path + hdf5_name, "r")
    for ID in filesToReadIdx:
        x = torch.tensor(file['data'][ID, :, :, :])
    file.close()
    print("End time: ", (time.time() - start_time))
####################################
n_samples = 105829
filesToReadIdx = random.sample(range(1, n_samples), batch_size)
print('Storing ', n_samples, ' samples wav in one large ndarray. But format: [samples,16000]')
list_ID = partition_path['train']
#print(len(list_ID))
start_time = time.time()
with h5py.File(datasethdf5_path + "allWavIdx.hdf5", "r") as file:
    for i in filesToReadIdx:
        x = torch.tensor(file['ds/data'][i, :])
print("End time: ", (time.time() - start_time))

#############################
filesToReadIdx = random.sample(range(1, 10000), batch_size)
print('Storing 10000 samples wav in one large ndarray. But format: [samples,16000]')
list_ID = partition_path['train']
#print(len(list_ID))
start_time = time.time()
with h5py.File(datasethdf5_path + "testWav1.hdf5", "r") as file:
    for i in filesToReadIdx:
        x = torch.tensor(file['ds/data'][i, :])
print("End time: ", (time.time() - start_time))

#############################
n_samples = 105829
filesToReadIdx = random.sample(range(1, n_samples), batch_size)
print('Storing 105829 samples wav in one large ndarray. But format: [samples,16000]')
list_ID = partition_path['train']
#print(len(list_ID))
start_time = time.time()
with h5py.File(datasethdf5_path + "testWav2.hdf5", "r") as file:
    for i in filesToReadIdx:
        x = torch.tensor(file['ds/data'][i, :])
print("End time: ", (time.time() - start_time))

#############################
n_samples = 105000
filesToReadIdx = random.sample(range(1, n_samples), batch_size)
print('Storing 105829 samples wav in one large ndarray. But format: [samples,16000]')
list_ID = partition_path['train']
#print(len(list_ID))
start_time = time.time()
with h5py.File(datasethdf5_path + "allWavIdx.hdf5", "r") as file:
    for i in filesToReadIdx:
        x = torch.tensor(file['ds/data'][i, :])
print("End time: ", (time.time() - start_time))

n_samples = 50000
filesToReadIdx = random.sample(range(1, n_samples), batch_size)
print('Storing 50000 samples wav in one large ndarray. But format: [samples,16000]')
list_ID = partition_path['train']
#print(len(list_ID))
start_time = time.time()
with h5py.File(datasethdf5_path + "testWav6.hdf5", "r") as file:
    for i in filesToReadIdx:
        x = torch.tensor(file['ds/data'][i, :])
print("End time: ", (time.time() - start_time))


n_samples = 85000
filesToReadIdx = random.sample(range(1, n_samples), batch_size)
print('Storing 85000 samples wav in one large ndarray. But format: [samples,16000]')
list_ID = partition_path['train']
#print(len(list_ID))
start_time = time.time()
with h5py.File(datasethdf5_path + "testWav7.hdf5", "r") as file:
    for i in filesToReadIdx:
        x = torch.tensor(file['ds/data'][i, :])
print("End time: ", (time.time() - start_time))

n_samples = 50000
filesToReadIdx = random.sample(range(1, n_samples), batch_size)
print('Storing 50000 samples wav in one large ndarray. But format: [samples,16000]')
list_ID = partition_path['train']
#print(len(list_ID))
start_time = time.time()
with h5py.File(datasethdf5_path + "testWav8.hdf5", "r") as file:
    for i in filesToReadIdx:
        x = torch.tensor(file['ds/data'][i, :])
print("End time: ", (time.time() - start_time))

n_samples = 105829
filesToReadIdx = random.sample(range(1, n_samples), batch_size)
print('Storing 105829 samples wav in one large ndarray. But format: [samples,16000]')
list_ID = partition_path['train']
#print(len(list_ID))
start_time = time.time()
with h5py.File(datasethdf5_path + "testWav9.hdf5", "r") as file:
    for i in filesToReadIdx:
        x = torch.tensor(file['ds/data'][i, :])
print("End time: ", (time.time() - start_time))

n_samples = 10000
filesToReadIdx = random.sample(range(1, n_samples), batch_size)
print('Storing 10000 samples wav in one large ndarray. But format: [samples,16000]')
list_ID = partition_path['train']
#print(len(list_ID))
start_time = time.time()
with h5py.File(datasethdf5_path + "testWav10.hdf5", "r") as file:
    time.sleep(1)
    for i in filesToReadIdx:
        x = torch.tensor(file['ds/data'][i, :])
print("End time: ", (time.time() - start_time - 1))



print('Finished')


