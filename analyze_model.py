# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 00:07:06 2019

@author: Brolof
"""

from cnn_model import ConvNet
from torchsummary import summary

model = ConvNet()
model.cuda()

summary(model, (1, 70, 99))
