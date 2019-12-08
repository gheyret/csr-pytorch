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
from torch.utils.data.sampler import Sampler
import csv
#from scipy.io import wavfile
#import soundfile as wavfile
import librosa
import numpy
import time
import matplotlib.pyplot as plt
from collections import OrderedDict

def normalize_spec(input_spec):
    mean = numpy.mean(input_spec)
    std = numpy.std(input_spec)

    output_spec = input_spec - mean
    output_spec = output_spec/std
    return output_spec

class Dataset(data.Dataset):

    def __init__(self, collection, num_features_input, use_delta_features=False, input_type='features'):
        '''

        :param collection: Key: Full path to file, value: true label sequence.
        ex:     {/path/to/file.wav: [0 1 2 3], ...}
        Note that if the dict is supposed to be fetched in sequence of raising length then it first needs to be ordered.

        :param list_ids: Expects list_IDs to be a list of file names for the individual wav files.
        :param wavfolder_path: The relative path to the folder where the raw wav files can be found.
        :param label_dict: A dictionary where the keys are the file names and the lists are the labels in ints. e.g.
                            label_dict["file_name_1.WAV"] = [3, 27, 33]
        '''
        'Initialization'
        self.collection = collection
        self.collection_items = list(self.collection.items())

        self.transformData = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.0], std=[0.5], inplace=True)])
        self.dataset = None
        self.nfft = 512
        self.nfilt = num_features_input
        self.window_size = 0.02  # s
        self.step_size = 0.01  # s
        self.samplerate = 16000
        self.input_type = input_type
        self.use_delta_features = use_delta_features

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.collection)

    def __getitem__(self, index):
        (path_to_file, true_label) = self.collection_items[index]
        y = true_label

        test_sound, samplerate = librosa.load(path_to_file, sr=self.samplerate)
        test_sound = numpy.trim_zeros(test_sound, 'b')
        if self.input_type == 'features':
            spec = librosa.feature.melspectrogram(y=test_sound, sr=samplerate, S=None, n_fft=self.nfft,
                                                   hop_length=int(self.step_size * samplerate),
                                                   win_length=int(self.window_size * samplerate), window='hamming',
                                                   center=False, pad_mode='reflect', power=1.0, n_mels=self.nfilt)
            spec = librosa.power_to_db(spec, ref=numpy.max)

            if self.use_delta_features:
                spec_delta = librosa.feature.delta(spec)
                spec_delta2 = librosa.feature.delta(spec, order=2)
                spec = normalize_spec(spec)
                spec_delta = normalize_spec(spec_delta)
                spec_delta2 = normalize_spec(spec_delta2)
                spec = numpy.array([spec, spec_delta, spec_delta2])
                #spec = numpy.concatenate((spec, spec_delta, spec_delta2), axis=0)

            x = self.transformData(spec).float()
            if self.use_delta_features:
                x = x.transpose(0, 1).transpose(1, 2)
        elif self.input_type is 'power':
            sound = test_sound.astype('float32') / 32767
            D = librosa.stft(sound, n_fft=self.nfft, hop_length=int(self.step_size * samplerate),
                             win_length=int(self.window_size * samplerate), window='hamming')
            spect, phase = librosa.magphase(D)
            spec = numpy.log1p(spect)
            x = torch.from_numpy(spec).float()
            x = x.unsqueeze(0)

        elif self.input_type == 'raw':
            #x = torch.from_numpy(test_sound).float()
            #x = x.unsqueeze(0).unsqueeze(0)
            x = numpy.expand_dims(test_sound, axis=0)
            x = self.transformData(x).float()


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
    minibatch_size = len(batch)
    num_features = longest_sample.size(0)
    freq_size = longest_sample.size(1)
    max_seqlength = longest_sample.size(2)
    inputs = torch.zeros(minibatch_size, num_features, freq_size, max_seqlength)

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

