
import h5py
import os
import torchvision.transforms as transforms
from scipy.io import wavfile
from Scripts.front_end_processing import logfbank
from Scripts.front_end_processing import normalizeSpectrogram
from Scripts.front_end_processing import padSpectrogram
import math
import numpy

dataset_path = "../data/"
hdf5_dir = "../input_data_hdf5_70m/"

transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.0],
                 std=[0.5])])

subFolderList = []
for x in os.listdir(dataset_path):
    if os.path.isdir(dataset_path + '/' + x):
        subFolderList.append(x)

test_ID = []
with open(dataset_path + "testing_list.txt", "r") as f:
    if f.mode == 'r':
        contents = f.readlines()

    print("number of test elements: %d" % len(contents))

    for s in contents:
        s = s.replace("\n", '')
        s = s.replace(".wav", '')
        test_ID.extend([s])

validation_ID = []
with open(dataset_path + "validation_list.txt", "r") as f:
        if f.mode == 'r':
            contents = f.readlines()

        print("number of validation elements: %d" % len(contents))

        for s in contents:
            s = s.replace("\n", '')
            s = s.replace(".wav", '')
            validation_ID.extend([s])

# Construct a dictionary containing all test/validation IDs
partition = {"test":test_ID}
partition["validation"] = validation_ID



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




nfft = 512
nfilt = 70
window_size = 0.02 # s
step_size = 0.01 # s
expectedRows = nfilt
expectedCols = 99

n_samples = len(test_ID)
n = 11
print(n)
hdf5_name = "testArraySize" + str(n) + ".hdf5"
print(hdf5_name)
n_samples = min(n*10000,len(data_ID))
if not os.path.exists(hdf5_dir + hdf5_name):
    # Generating a hdf5 file with a single ndarray [index,1,70,99] in group 'data'
    print('Starting transformation process:')
    with h5py.File(hdf5_dir + hdf5_name, "w") as file:
        dsetData = file.create_dataset('data', (n_samples, 1, 70, 99))
        dsetLabel = file.create_dataset('label', (n_samples,), dtype='i8')
        for i, ID in enumerate(data_ID[0:n_samples], 0):
            samplerate, test_sound = wavfile.read('./data/' + ID + ".wav")
            # print(ID)
            spec = logfbank(test_sound, samplerate,
                            winlen=window_size,
                            winstep=step_size,
                            nfilt=nfilt,
                            nfft=nfft)
            spec = normalizeSpectrogram(spec.T)
            spec = padSpectrogram(spec, expectedRows, expectedCols)
            file_name = ID.replace('.wav', '')
            tensor = transform(spec).float().long()

            # torch.save(tensor, output_path + file_name + ".pt")
            dsetData[i, :, :, :] = tensor
            dsetLabel[i] = index_label_ID_table[ID.split("/")[0]]

            if (i + 1) % (n_samples / 20) == 0:
                print("{0:.3f}% completed".format(i / n_samples * 100.0))
        # numpy.savetxt(output_path + file_name + ".png", spec, delimiter=",")

        print("100.000% completed")
        print("Finished processing data")


# Generating a hdf5 file with waveforms

expectedRows = 1
expectedCols = 99
n_samples = len(data_ID)

if not os.path.exists(hdf5_dir + "allWavIdx.hdf5"):
    # Generating a hdf5 file with a single ndarray [index,1,70,99] in group 'data'
    print('Starting transformation process:')
    with h5py.File(hdf5_dir + "allWavIdx.hdf5", "w") as file:
        dsetData = file.create_dataset('ds/data', (n_samples, 16000))
        dsetLabel = file.create_dataset('ds/label', (n_samples,), dtype = 'i8')
        for i, ID in enumerate(data_ID[0:n_samples], 0):
            print(i)
            samplerate, test_sound = wavfile.read('../data/' + ID + ".wav")
            z = numpy.zeros((1, 16000))
            z[0, :test_sound.shape[0]]=test_sound
            file_name = ID.replace('.wav', '')
            dsetData[i, :] = z
            dsetLabel[i] = index_label_ID_table[ID.split("/")[0]]

            if (i + 1) % (n_samples / 100) == 0:
                print("{0:.3f}% completed".format(i / n_samples * 100.0))

        print("100.000% completed")
        print("Finished processing data")


n_samples = len(data_ID)
if not os.path.exists(hdf5_dir + "allWavIdxNoPad.hdf5"):
    # Generating a hdf5 file with a single ndarray [index,1,70,99] in group 'data'
    print('Starting transformation process:')
    with h5py.File(hdf5_dir + "allWavIdxNoPad.hdf5", "w") as file:
        dsetData = file.create_dataset('ds/data', (n_samples, 16000))
        dsetLabel = file.create_dataset('ds/label', (n_samples,), dtype = 'i8')
        for i, ID in enumerate(data_ID[0:n_samples], 0):
            samplerate, test_sound = wavfile.read('../data/' + ID + ".wav")
            file_name = ID.replace('.wav', '')
            dsetData[i, :] = test_sound
            dsetLabel[i] = index_label_ID_table[ID.split("/")[0]]
            if (i + 1) % (n_samples / 100) == 0:
                print("{0:.3f}% completed".format(i / n_samples * 100.0))

        print("100.000% completed")
        print("Finished processing data")


n_samples = 10000#len(data_ID)

if not os.path.exists(hdf5_dir + "testWav1.hdf5"):
    # Generating a hdf5 file with a single ndarray [index,1,70,99] in group 'data'
    arrayData = numpy.zeros((n_samples,16000), dtype=numpy.int16)
    arrayLabels = numpy.zeros((n_samples,) , dtype=numpy.int16)
    print('Starting transformation process:')
    for i, ID in enumerate(data_ID[0:n_samples], 0):
        samplerate, test_sound = wavfile.read('../data/' + ID + ".wav")
        z = numpy.zeros((1,16000))
        z[0,:test_sound.shape[0]]=test_sound
        file_name = ID.replace('.wav', '')
        arrayData[i,:] = z
        arrayLabels[i] = index_label_ID_table[ID.split("/")[0]]
        if (i + 1) % (n_samples / 100) == 0:
            print("{0:.3f}% completed".format(i / n_samples * 100.0))
    with h5py.File(hdf5_dir + "testWav1.hdf5", "w") as file:
        dsetData = file.create_dataset('ds/data', (n_samples,16000))
        dsetLabel = file.create_dataset('ds/label', (n_samples,), dtype='i8')
        dsetData[:, :] = arrayData
        dsetLabel[:] = arrayLabels

# Test chunking
n_samples = 10000
if not os.path.exists(hdf5_dir + "testChunking.hdf5"):
    # Generating a hdf5 file with a single ndarray [index,1,70,99] in group 'data'
    arrayData = numpy.zeros((n_samples, 16000), dtype=numpy.int16)
    arrayLabels = numpy.zeros((n_samples,), dtype=numpy.int16)
    print('Starting transformation process:')
    for i, ID in enumerate(data_ID[0:n_samples], 0):
        samplerate, test_sound = wavfile.read('../data/' + ID + ".wav")
        z = numpy.zeros((1, 16000))
        z[0, :test_sound.shape[0]] = test_sound
        file_name = ID.replace('.wav', '')
        arrayData[i, :] = z
        arrayLabels[i] = index_label_ID_table[ID.split("/")[0]]
        if (i + 1) % (n_samples / 20) == 0:
            print("{0:.3f}% completed".format(i / n_samples * 100.0))
if not os.path.exists(hdf5_dir + "testChunking.hdf5"):
    with h5py.File(hdf5_dir + "testChunking.hdf5", "w") as file:
        dsetData = file.create_dataset('ds/data', (n_samples, 16000), chunks = (1,16000))
        dsetLabel = file.create_dataset('ds/label', (n_samples,), dtype='i8')
        dsetData[:, :] = arrayData
        dsetLabel[:] = arrayLabels
if not os.path.exists(hdf5_dir + "testChunking2.hdf5"):
    with h5py.File(hdf5_dir + "testChunking2.hdf5", "w") as file:
        dsetData = file.create_dataset('ds/data', (n_samples, 16000), chunks = (5,16000))
        dsetLabel = file.create_dataset('ds/label', (n_samples,), dtype='i8')
        dsetData[:, :] = arrayData
        dsetLabel[:] = arrayLabels
if not os.path.exists(hdf5_dir + "testChunking3.hdf5"):
    with h5py.File(hdf5_dir + "testChunking3.hdf5", "w") as file:
        dsetData = file.create_dataset('ds/data', (n_samples, 16000), chunks = (100,16000))
        dsetLabel = file.create_dataset('ds/label', (n_samples,), dtype='i8')
        dsetData[:, :] = arrayData
        dsetLabel[:] = arrayLabels
if not os.path.exists(hdf5_dir + "testChunking4.hdf5"):
    with h5py.File(hdf5_dir + "testChunking4.hdf5", "w") as file:
        dsetData = file.create_dataset('ds/data', (n_samples, 16000), chunks=(1000, 16000))
        dsetLabel = file.create_dataset('ds/label', (n_samples,), dtype='i8')
        dsetData[:, :] = arrayData
        dsetLabel[:] = arrayLabels
if not os.path.exists(hdf5_dir + "testChunking5.hdf5"):
    with h5py.File(hdf5_dir + "testChunking5.hdf5", "w") as file:
        dsetData = file.create_dataset('ds/data', (n_samples, 16000), chunks=(5000, 16000))
        dsetLabel = file.create_dataset('ds/label', (n_samples,), dtype='i8')
        dsetData[:, :] = arrayData
        dsetLabel[:] = arrayLabels
if not os.path.exists(hdf5_dir + "testChunking6.hdf5"):
    with h5py.File(hdf5_dir + "testChunking6.hdf5", "w") as file:
        dsetData = file.create_dataset('ds/data', (n_samples, 16000), chunks=(10000, 16000))
        dsetLabel = file.create_dataset('ds/label', (n_samples,), dtype='i8')
        dsetData[:, :] = arrayData
        dsetLabel[:] = arrayLabels


    print("100.000% completed")
    print("Finished processing data")

