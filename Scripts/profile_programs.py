# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:36:34 2019

@author: Brolof
"""

import cProfile
#from train_model import trainModel
import train_model
from train_model import runMultipleThreads


# Anaconda prompt:
# snakeviz E:\Documents\1.Chalmers\Examensarbete\Code\GoogleSpeechCommands\train_model.prof

cProfile.run('runMultipleThreads()', 'train_model_MT.prof')

print("Finished")


#%%

import cProfile
from train_model import trainModel


# Anaconda prompt:
# snakeviz E:\Documents\1.Chalmers\Examensarbete\Code\GoogleSpeechCommands\train_model.prof
# pyprof2calltree -k -i E:\Documents\1.Chalmers\Examensarbete\Code\GoogleSpeechCommands\train_model.prof

cProfile.run('trainModel()', 'train_model.out')

print("Finished")