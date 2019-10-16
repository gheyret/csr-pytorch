# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:13:24 2019

@author: Brolof
"""
import torch.nn as nn


class ConvNet(nn.Module):
    # Todo: Should have wider kernel so that it looks more before and after
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=[7, 3], stride=1, padding=[3, 1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[2, 1], stride=[2, 1]))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=[7, 1], stride=1, padding=[3, 0]),
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
        #print(x.size())
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