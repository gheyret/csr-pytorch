# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:22:05 2019

@author: Brolof
"""
import os
import numpy
import csv


def csvToList(path_to_csv):
    with open(path_to_csv, newline='') as csvfile:
        output_list = list(csv.reader(csvfile))
        output_list = numpy.asarray(output_list)[0]
        output_list = list(map(int, output_list))
        return output_list


def load_data_set_path(dataset_path):
    # VERBOSE = False
    partition, labels, label_index_ID_table = importData(dataset_path, 35)  # IDs
    return partition, labels


def load_data_set_indexes(dataset_path_in):
    partition_out = {'train': csvToList(dataset_path_in + "train_idx.csv"),
                     'validation': csvToList(dataset_path_in + "validation_idx.csv"),
                     'test': csvToList(dataset_path_in + "test_idx.csv")}

    return partition_out


def importData(dataset_path, n_words=35, n_samples=105829, VERBOSE=False):
    # Read the testing file and extract all IDs
    test_ID = []
    f = open(dataset_path + "testing_list.txt", "r")
    if f.mode == 'r':
        contents = f.readlines()

    if VERBOSE: print("number of test elements: %d" % len(contents))

    for s in contents:
        s = s.replace("\n", '')
        s = s.replace(".wav", '')
        test_ID.extend([s])

    # Read the validation file and extract all IDs
    validation_ID = []
    f = open(dataset_path + "validation_list.txt", "r")
    if f.mode == 'r':
        contents = f.readlines()

    if VERBOSE: print("number of validation elements: %d" % len(contents))

    for s in contents:
        s = s.replace("\n", '')
        s = s.replace(".wav", '')
        validation_ID.extend([s])

        # Construct a dictionary containing all test/validation IDs
    partition = {"test": test_ID, "validation": validation_ID}

    # Extract ID of all files and store Label & Id in categories train/test/validation
    subFolderList = []
    for x in os.listdir(dataset_path):
        if os.path.isdir(dataset_path + '/' + x):
            subFolderList.append(x)

            # n_words = 35

    # print(subFolderList[0:n_words]) #(subFolderList[0:n_words])
    data_ID = []
    label_ID = []
    total = 0
    label_index = 0
    label_index_ID_table = {}
    for x in (subFolderList[0:n_words]):  # (subFolderList[0:n_words])
        # get all the wave files
        all_files = [x + '/' + y for y in os.listdir(dataset_path + x) if '.wav' in y]
        total += len(all_files)

        all_files = [z.replace(".wav", '') for z in all_files]  # remove .wav

        data_ID += all_files
        label_ID += ([label_index] * len(all_files))
        label_index_ID_table[label_index] = x
        # show file counts
        label_index += 1
        if VERBOSE: print('count: %d : %s' % (len(all_files), x))
    if VERBOSE: print("Total number of files: ", total)

    labels = dict(zip(data_ID, label_ID))

    prohibited_ID = set(test_ID + validation_ID)
    train_ID = [x for x in data_ID if x not in prohibited_ID]

    partition["train"] = train_ID

    if VERBOSE: print("Sorted into categories:")
    for k, v in partition.items():
        if VERBOSE: print(k, len(v))

    return partition, labels, label_index_ID_table
