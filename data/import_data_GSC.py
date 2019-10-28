# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:22:05 2019

@author: Brolof
"""
import os
import numpy
import csv


def csv_to_list(path_to_csv):
    with open(path_to_csv) as csvfile: # , newline='' in Windows
        output_list = list(csv.reader(csvfile))
        output_list = numpy.asarray(output_list)[0]
        output_list = list(map(int, output_list))
        return output_list


def load_data_set_indexes(dataset_path_in):
    partition_out = {'train': csv_to_list(dataset_path_in + "train_idx.csv"),
                     'validation': csv_to_list(dataset_path_in + "validation_idx.csv"),
                     'test': csv_to_list(dataset_path_in + "test_idx.csv")}

    return partition_out


def import_data(dataset_path, n_words=35, verbose=False, return_index=False):
    # Read the testing file and extract all IDs
    test_id = []
    f = open(dataset_path + "testing_list.txt", "r")
    if f.mode == 'r':
        contents = f.readlines()

    if verbose:
        print("number of test elements: %d" % len(contents))

    for s in contents:
        s = s.replace("\n", '')
        s = s.replace(".wav", '')
        test_id.extend([s])

    # Read the validation file and extract all IDs
    validation_ID = []
    f = open(dataset_path + "validation_list.txt", "r")
    if f.mode == 'r':
        contents = f.readlines()

    if verbose: print("number of validation elements: %d" % len(contents))

    for s in contents:
        s = s.replace("\n", '')
        s = s.replace(".wav", '')
        validation_ID.extend([s])

        # Construct a dictionary containing all test/validation IDs
    partition = {"test": test_id, "validation": validation_ID}

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
        if verbose: print('count: %d : %s' % (len(all_files), x))
    if verbose: print("Total number of files: ", total)

    labels = dict(zip(data_ID, label_ID))

    prohibited_ID = set(test_id + validation_ID)
    train_ID = [x for x in data_ID if x not in prohibited_ID]

    partition["train"] = train_ID

    if verbose: print("Sorted into categories:")
    for k, v in partition.items():
        if verbose: print(k, len(v))


    if return_index:
        train_idx = []
        validation_idx = []
        test_idx = []
        n_samples = len(data_ID)
        for i, ID in enumerate(data_ID):
            if (i) % (n_samples/20) == 0:
                print("{0:.3f}% completed".format(i/n_samples*100.0))
            if ID in train_ID:
                train_idx.append(i)
            elif ID in validation_ID:
                validation_idx.append(i)
            elif ID in test_id:
                test_idx.append(i)
        partition_idx = {"train":train_idx, "validation":validation_idx, "test":test_idx}
        return partition_idx, labels, label_index_ID_table

    return partition, labels, label_index_ID_table
