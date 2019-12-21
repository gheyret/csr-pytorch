# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 00:07:06 2019

@author: Brolof
"""
from models.FuncNet1 import FuncNet1
from models.FuncNet2 import FuncNet2
from models.FuncNet3 import FuncNet3
from models.FuncNet4 import FuncNet4
from models.ConvNet3 import ConvNet3
from models.ConvNet4 import ConvNet4
from models.ConvNet5 import ConvNet5
from models.ConvNet6 import ConvNet6
from models.ConvNet7 import ConvNet7
from models.ConvNet8 import ConvNet8
from models.ConvNet9 import ConvNet9
from models.ConvNet10 import ConvNet10
from models.ConvNet11 import ConvNet11
from models.ConvNet12 import ConvNet12
from models.ConvNet13 import ConvNet13
from models.ConvNet14 import ConvNet14
from models.ConvNet15 import ConvNet15
from models.DNet1 import DNet1
from models.DNet2 import DNet2
from models.RawNet2 import RawNet2

from analytics.torchsummary import summary

#model = Net()
for i_model, model in enumerate([ConvNet3(), ConvNet4(), ConvNet5(), ConvNet6(), ConvNet7(), ConvNet8(),
                                 ConvNet9(), ConvNet10(), ConvNet11(), ConvNet12(), ConvNet13(), ConvNet14(), ConvNet15()]):
    print("--------------------------------------------------------------------ConvNet" + str(i_model + 2))
    model.cuda()
    summary(model, (1, 70, 1600))
    print("###############################################")

model = DNet1()
model.cuda()
summary(model, (3, 70, 1600))

model = RawNet2()
model.cuda()
summary(model, (1, 1, 16000))

model = FuncNet1(num_features_input=40, num_input_channels=3)
model.cuda()
summary(model, (3, 40, 1600))

model = FuncNet2(num_features_input=40, num_input_channels=3)
model.cuda()
summary(model, (3, 40, 1600))

model = FuncNet3(num_features_input=40, num_input_channels=3)
model.cuda()
summary(model, (3, 40, 1600))

model = FuncNet4(num_features_input=120, num_input_channels=1)
model.cuda()
summary(model, (1, 120, 1600))