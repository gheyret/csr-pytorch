# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 00:07:06 2019

@author: Brolof
"""

from cnn_model import ConvNet2 as Net
from torchsummary import summary

model = Net()
model.cuda()

summary(model, (1, 70, 99))
