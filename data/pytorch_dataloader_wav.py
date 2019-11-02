# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:17:39 2019

"""
import h5py
import torch
from torch.utils import data
from torchvision.transforms import transforms
from data.front_end_processing import logfbank
from torch.utils.data import DataLoader
import csv
#from scipy.io import wavfile
#import soundfile as wavfile
import librosa
import numpy
import time
import matplotlib.pyplot as plt



class Dataset(data.Dataset):

    def __init__(self, list_ids, wavfolder_path, label_dict):
        '''

        :param list_ids: Expects list_IDs to be a list of file names for the individual wav files.
        :param wavfolder_path: The relative path to the folder where the raw wav files can be found.
        :param label_dict: A dictionary where the keys are the file names and the lists are the labels in ints. e.g.
                            label_dict["file_name_1.WAV"] = [3, 27, 33]
        '''
        'Initialization'
        self.list_ids = list_ids
        self.label_dict = label_dict
        self.wavfolder_path = wavfolder_path
        self.transformData = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.0],
                                  std=[0.5])])
        self.dataset = None
        self.nfft = 512
        self.nfilt = 70
        self.window_size = 0.02  # s
        self.step_size = 0.01  # s
        self.samplerate = 16000#22050

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_ids)

    def __getitem__(self, index):
        file_name = self.list_ids[index]

        # ID contains the path to the wav file
        wav_path = self.wavfolder_path + file_name
        test_sound, samplerate = librosa.load(wav_path, sr=self.samplerate)
        #Todo: Make sure it's upsampled to 22050 correctly
        #self.samplerate = samplerate
        y = self.label_dict[file_name]

        test_sound = numpy.trim_zeros(test_sound, 'b')
        spec = logfbank(test_sound, self.samplerate,
                        winlen=self.window_size,
                        winstep=self.step_size,
                        nfilt=self.nfilt,
                        nfft=self.nfft)

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


def collate_fn_pad(batch):
    '''
    Pads batch of variable length
    '''

    ## get sequence lengths
    def funcTensor(p):
        return p[0].size(2)

    def funcTarget(p):
        return len(p[1])

    batch = sorted(batch, key=lambda sample: sample[0].size(2), reverse=True)
    longest_sample = max(batch, key=funcTensor)[0]
    freq_size = longest_sample.size(1)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(2)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)

    longest_target = max(batch, key=funcTarget)[1]
    max_targetlength = len(longest_target)
    targets = torch.zeros(minibatch_size, max_targetlength, dtype=torch.int64)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    #targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(2)
        inputs[x].narrow(2, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets[x].narrow(0, 0, len(target)).copy_(torch.IntTensor(target))
    return inputs, targets, input_percentages, target_sizes


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = collate_fn_pad
