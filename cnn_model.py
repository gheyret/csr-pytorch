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
            nn.Conv1d(num_features, 250, 41, 1, 20),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(250, 250, 7, 1, 3),
            nn.ReLU(),
            nn.Conv1d(250, 250, 7, 1, 3),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(250, 2000, 31, 1, 15),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(2000, 2000, 1),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(2000, num_classes, 1)
        )
        self.softMax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Input on Form: N x 1 x F x T
        # Output on Form: T x N x C for CTC_loss
        # N = Batch size, F = Features, T = Time, C = Classes

        N, _, F, T = x.size()
        x = x.view(N, F, T)

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
        N, C, T = out.size()
        out = out.view(T, N, C) # Ordering for CTCLoss TxNxC
        return out


class ConvNet2(nn.Module):
    # Todo: Should have wider kernel so that it looks more before and after
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=[41, 11], stride=(1, 2), padding=[0, 5]),
            nn.BatchNorm2d(64),
            nn.Hardtanh(0, 20, inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=[11, 7], stride=1, padding=[0, 3]),
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
        # self.globalMaxPool = nn.MaxPool2d()
        # self.drop_out = nn.Dropout()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),  #
            nn.ReLU(),
            nn.Linear(256, 46))
        self.softMax = nn.LogSoftmax(dim=-1)

    def forward(self, x, input_lengths):
        #Todo: Clean up
        N, _, F, T_in = x.size()
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        N, FM, F, T = out.size() # N = Batch size, FM = Feature maps, f = frequencies, t = time
        out = out.view(N, T, F, FM)
        # FC: connect a tensor(N, *, in_features) to (N, *, out_features)
        out = self.fc(out) # NxTx1xC
        out = self.softMax(out)

        N, T_out, _, C  = out.size()
        out = out.view(T_out, N, C) # Ordering for CTCLoss TxNxC

        time_refactoring = T_out/T_in
        output_lengths = torch.floor(input_lengths.float().mul_(time_refactoring)).int()

        return out, output_lengths
