# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 00:07:06 2019

@author: Brolof
"""
from models.ConvNet2 import ConvNet2
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


from analytics.torchsummary import summary

#model = Net()
for i_model, model in enumerate([ConvNet2(), ConvNet3(), ConvNet4(), ConvNet5(), ConvNet6(), ConvNet7(), ConvNet8(),
                                 ConvNet9(), ConvNet10(), ConvNet11(), ConvNet12()]):
    print("--------------------------------------------------------------------ConvNet" + str(i_model + 2))
    model.cuda()
    summary(model, (1, 70, 1600))
    print("###############################################")
