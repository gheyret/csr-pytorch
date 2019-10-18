# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:17:39 2019

@author: Brolof
"""
import h5py
import torch
from torch.utils import data
from torchvision.transforms import transforms
from Scripts.front_end_processing import logfbank
from torch.utils.data import DataLoader
import csv

import numpy
import time
import matplotlib.pyplot as plt
#Todo: Make new Encoder for generated dataset
class GoogleSpeechEncoder:
    def __init__(self):
        self.label_list = ['_', ' ', 'AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'EHR', 'ER', 'EY', 'F',
                           'G', 'H', 'IH', 'IY', 'IYR', 'JH', 'K', 'L', 'M', 'N', 'NG', 'O', 'OW', 'OY', 'P', 'R', 'S',
                           'SH', 'T', 'TH', 'UH', 'UHR', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'] # 0 = blank
        label_dict = dict()
        for i, x in enumerate(self.label_list):
            label_dict[x] = i
        self.label_dict = label_dict

        path_to_csv = "../data/GoogleSpeechCommands/hdf5_format/label_index_ID_table.csv"
        with open(path_to_csv, newline='') as csvfile:
            output_list = list(csv.reader(csvfile))
        output_list = [x for x in output_list if x]
        self.output_dict = dict(output_list)

        self.switcher = {
            0: "B AE K W AX D", #backward
            1: "B EH D",       #bed
            2: "B ER D",       #bird
            3: "K AE T",       #cat
            4: "D O G",       #dog
            5: "D AW N",       #down
            6: "EY T",         #'eight',
            7: "F AY V",       #'five',
            8: "F O L OW",     #'follow',
            9: "F AO W AX D",  #'forward',
            10: "F AO",        #'four',
            11: "G OW",        #'go',
            12: "H AE P IY",   #'happy',
            13: "H AW S",      #'house',
            14: "L ER N",            #'learn',
            15: "L EH F T",            #'left',
            16: "M AA V IH N",            #'marvin',
            17: "N AY N",            #'nine',
            18: "N OW",            #'no',
            19: "O F",            #'off',
            20: "O N",            #'on',
            21: "W AH N",            #'one',
            22: "R AY T",            #'right',
            23: "S EH V AX N",            #'seven',
            24: "SH IY L AX",            #'sheila',
            25: "S IH K S",            #'six',
            26: "S T O P",            #'stop',
            27: "TH R IY",            #'three',
            28: "T R IY",            # 'tree',
            29: "T UW",            #'two',
            30: "AH P",            #'up',
            31: "V IH ZH UHR L",            #'visual',
            32: "W AW",             #'wow',
            33: "Y EH S",            #'yes',
            34: "Z IYR R OW"            #'zero'
            }

    def encode_labels(self, word_ID):
        phonetic_str = self.switcher.get(word_ID)
        #print(phonetic_str)
        phonetic_list = phonetic_str.split(" ")
        phonetic_id_list = [self.label_dict[x] for x in phonetic_list]
        return phonetic_id_list

    def decode_codes(self):
        raise NotImplementedError

class Dataset(data.Dataset, GoogleSpeechEncoder):

    def __init__(self, hdf5file_path, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs
        self.hdf5file_path = hdf5file_path
        self.transformData = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.0],
                                  std=[0.5])])
        self.dataset = None
        self.nfft = 512
        self.nfilt = 70
        self.window_size = 0.02  # s
        self.step_size = 0.01  # s
        self.expectedRows = self.nfilt
        self.expectedCols = 99
        self.samplerate = 16000
        GoogleSpeechEncoder.__init__(self)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.hdf5file_path, "r", swmr=True) # SWMR = Single write multiple read
        #index = numpy.random.randint(1,10)
        ID = self.list_IDs[index]
        y = self.dataset['ds/label'][ID]
        y = self.encode_labels(y)
        test_sound = self.dataset['ds/data'][ID, :]
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
    Deep speech
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
