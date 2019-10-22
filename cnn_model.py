# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:13:24 2019

@author: Brolof
"""
import torch.nn as nn
import torch


class ConvNet(nn.Module):
    # Todo: Should have wider kernel so that it looks more before and after
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=[21, 3], stride=1, padding=[10, 1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[2, 1], stride=[2, 1]))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=[11, 1], stride=1, padding=[5, 0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 1], stride=[3, 1]))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=[11, 1], stride=1, padding=[0, 0]),  # 256
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=[7, 1], stride=1, padding=[3, 0]),  # 512
            nn.ReLU())
        # self.globalMaxPool = nn.MaxPool2d()
        # self.drop_out = nn.Dropout()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),  #
            nn.ReLU(),
            nn.Linear(256, 46))
        self.softMax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        #Todo: Clean up
        print(x.size())
        out = self.layer1(x)
        #print("Size from first layer: ", out.size())
        out = self.layer2(out)
        #print("Size from second layer: ", out.size())
        out = self.layer3(out)
        #print("Size from 3rd layer: ", out.size())
        out = self.layer4(out)
        #print("Size from 4th layer: ", out.size())

        # Apply global max pool:
        #out = nn.functional.max_pool2d(out, kernel_size=out.size()[2:])
        #print("Size from 5th layer: ", out.size())

        #out = out.reshape(out.size(0), -1)
        #print("Size from second layer reshaped: ", out.size())

        #out = self.drop_out(out)
        n, fm, f, t = out.size() # N = Batch size, FM = Feature maps, f = frequencies, t = time
        out = out.view(n, t, f, fm)
        #print("Size from tensor reshaped: ", out.size())

        # connect a tensor(N, *, in_features) to (N, *, out_features)
        out = self.fc(out) # NxTx1xC
        out = self.softMax(out)
        # Todo: Perhaps check if the model is currently training, whether softmax should be applied or not

        batch_size, frames, f, classes = out.size() # N = Batch size, FM = Feature maps, f = frequencies, t = time
        out = out.view(frames,batch_size,classes) # Ordering for CTCLoss TxNxC
        #print("Size from 5th layer: ", out.size())


        return out

#50,16,20
#T, N, C
#Seq length, Batch size, Classes


class CTC_CNN(nn.Module):
    # Wav2Letter
    def __init__(self, num_features = 70, num_classes = 46):
        super(CTC_CNN, self).__init__()
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.layer1 = nn.Sequential(
            nn.Conv1d(num_features, 250, 25, 2),
            nn.BatchNorm1d(250),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(250, 250, 3),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Conv1d(250, 250, 3),
            nn.BatchNorm1d(250),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(250, 2000, 7),
            nn.BatchNorm1d(2000),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(2000, 2000, 1),
            nn.BatchNorm1d(2000),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(2000, num_classes, 1)
        )
        self.softMax = nn.LogSoftmax(dim=1)

    def forward(self, x, input_lengths = 0):
        # Input on Form: N x 1 x F x T
        # Output on Form: T x N x C for CTC_loss
        # N = Batch size, F = Features, T = Time, C = Classes

        N, _, F, T_in = x.size()
        x = x.view(N, F, T_in)

        # Layer 1 expects N x F x T
        #print("Size of input : ", x.size())
        out = self.layer1(x)
        #print("Size from first layer: ", out.size())
        out = self.layer2(out)
        #print("Size from second layer: ", out.size())
        out = self.layer3(out)
        #print("Size from third layer: ", out.size())
        out = self.layer4(out)
        #print("Size from fourth layer: ", out.size())
        out = self.layer5(out)
        #print("Size from fifth layer: ", out.size())
        out = self.softMax(out)
        N, C, T_out = out.size()
        out = out.view(T_out, N, C) # Ordering for CTCLoss TxNxC
        output_lengths = input_lengths - (T_in-T_out)
        return out, output_lengths


class ConvNet2(nn.Module):
    # Todo: Should have wider kernel so that it looks more before and after
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=[41, 11], stride=(1, 2), padding=[0, 5]),
            nn.BatchNorm2d(64),
            nn.Hardtanh(0, 20, inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=[11, 7], stride=(1,2), padding=[0, 3]),
            nn.BatchNorm2d(128),
            nn.Hardtanh(0, 20, inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=[11, 5], stride=1, padding=[0, 2]),  # 256
            nn.BatchNorm2d(256),
            nn.Hardtanh(0, 20, inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=[10, 5], stride=1, padding=[0, 2]),  # 512
            nn.BatchNorm2d(512),
            nn.Hardtanh(0, 20, inplace=True))

        self.rnn = nn.Sequential(
            nn.LSTM(input_size=512, hidden_size=512, num_layers=1, bidirectional=True, batch_first=False))
        # self.globalMaxPool = nn.MaxPool2d()
        # self.drop_out = nn.Dropout()
        self.fc = nn.Sequential(
            nn.Linear(512, 46))
        self.softMax = nn.LogSoftmax(dim=-1)

    def forward(self, x, input_lengths):
        #Todo: Clean up
        N, _, F, T_in = x.size()
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        N, FM, F, T_out = out.size() # N = Batch size, FM = Feature maps, f = frequencies, t = time

        time_refactoring = T_out / T_in

        output_lengths = torch.floor(input_lengths.float().mul_(time_refactoring)).int()

        out = out.view(N, FM*F, T_out)
        out = out.transpose(1, 2).transpose(0, 1).contiguous() # T x N x (F*FM)
        for i in range(0, 1):
            out = nn.utils.rnn.pack_padded_sequence(out, output_lengths, batch_first=False)
            out, h = self.rnn(out)
            out, _ = nn.utils.rnn.pad_packed_sequence(out)
            out = out.view(out.size(0), out.size(1), 2, -1).sum(2).view(out.size(0), out.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        # FC: connect a tensor(N, *, in_features) to (N, *, out_features)
        out = self.fc(out) # TxNxC
        out = self.softMax(out)
        return out, output_lengths


class RNN(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_dim, num_layers=n_layers, batch_first=False)

    def forward(self, x, hidden):
        # input of shape (seq_len, batch, input_size)
        # output of shape (seq_len, batch, num_directions * hidden_size)
        batch_size = x.size(1)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        #out = out.contiguous().view(-1, self.hidden_dim)
        #out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden