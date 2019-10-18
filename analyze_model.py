# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 00:07:06 2019

@author: Brolof
"""

from cnn_model import ConvNet2
from torchsummary import summary

model = ConvNet2()
model.cuda()

summary(model, (1, 70, 99))
