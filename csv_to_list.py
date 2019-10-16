# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:54:13 2019

@author: Brolof
"""

import csv
import numpy

def csvToList(path_to_csv):
    with open(path_to_csv, newline='') as csvfile:
        output_list = list(csv.reader(csvfile))
        output_list = numpy.asarray(output_list)[0]
        output_list = list(map(int, output_list))
        return output_list