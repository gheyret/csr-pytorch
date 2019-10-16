# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:13:24 2019

@author: Brolof
"""
import torch.nn as nn


class ConvNet(nn.Module):
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
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(512, 256)  #
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(256, 35)

    def forward(self, x):
        # print(x.size())
        out = self.layer1(x)
        # print("Size from first layer: ", out.size())
        out = self.layer2(out)
        # print("Size from second layer: ", out.size())
        out = self.layer3(out)
        # print("Size from 3rd layer: ", out.size())
        out = self.layer4(out)
        # print("Size from 4th layer: ", out.size())

        out = nn.functional.max_pool2d(out, kernel_size=out.size()[2:])
        # print("Size from 5th layer: ", out.size())

        out = out.reshape(out.size(0), -1)
        # print("Size from second layer reshaped: ", out.size())
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        return out
