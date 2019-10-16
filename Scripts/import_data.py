# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:22:05 2019

@author: Brolof
"""
import os

class ImportData:
#dataset_path = "./data/"

# training_ID
# validation_ID
# labels 
# partition =  {train: [ID1,ID2,...], valdiation : [ID3,ID4,...], test: [ID5,ID6,...]}
# labels = {ID1 : label1, ID2: label1, ID3: label2, ...}

    def importData(dataset_path, n_words = 35, n_samples = 105829, VERBOSE = False):
        # Read the testing file and extract all IDs
        test_ID = []
        f = open(dataset_path + "testing_list.txt", "r")
        if f.mode == 'r':
            contents = f.readlines()
            
        if VERBOSE: print("number of test elements: %d" % len(contents))
        
        for s in contents:
            s = s.replace("\n",'')
            s = s.replace(".wav",'')
            test_ID.extend([s])
            
            
        # Read the validation file and extract all IDs
        validation_ID = []
        f = open(dataset_path + "validation_list.txt", "r")
        if f.mode == 'r':
            contents = f.readlines()
            
        if VERBOSE: print("number of validation elements: %d" % len(contents))
        
        for s in contents:
            s = s.replace("\n",'')
            s = s.replace(".wav",'')
            validation_ID.extend([s])   
        
        
        # Construct a dictionary containing all test/validation IDs
        partition = {"test":test_ID}
        partition["validation"] = validation_ID
            

        # Extract ID of all files and store Label & Id in categories train/test/validation
        subFolderList = []
        for x in os.listdir(dataset_path):
            if os.path.isdir(dataset_path + '/' + x):
                subFolderList.append(x)   
            
        #n_words = 35 

        #print(subFolderList[0:n_words]) #(subFolderList[0:n_words])
        data_ID = []
        label_ID = []
        total = 0
        label_index = 0
        label_index_ID_table = {}
        for x in (subFolderList[0:n_words]): #(subFolderList[0:n_words])
            # get all the wave files
            all_files = [x + '/' + y for y in os.listdir(dataset_path + x) if '.wav' in y]
            total += len(all_files)
            
            all_files = [z.replace(".wav",'') for z in all_files] # remove .wav
            
            data_ID += all_files
            label_ID += ([label_index] * len(all_files))
            label_index_ID_table[label_index] = x
            # show file counts
            label_index += 1
            if VERBOSE: print('count: %d : %s' % (len(all_files), x ))
        if VERBOSE: print("Total number of files: ",total)
        
        labels = dict(zip(data_ID,label_ID))
        
        prohibited_ID = set(test_ID + validation_ID)
        train_ID = [x for x in data_ID if x not in prohibited_ID]
        
        partition["train"] = train_ID
        
        if VERBOSE: print("Sorted into categories:")
        for k, v in partition.items():
            if VERBOSE: print(k, len(v))
            
            
            
        # train_idx = []
        # validation_idx = []
        # test_idx = []
        # for i, ID in enumerate(data_ID[0:n_samples],0):
        #     if (i) % (n_samples/20) == 0:
        #         print("{0:.3f}% completed".format(i/n_samples*100.0))
        #     if ID in train_ID:
        #         train_idx.append(i)
        #     elif ID in validation_ID:
        #         validation_idx.append(i)
        #     elif ID in test_ID:
        #         test_idx.append(i)
        # partition_idx = {"train":train_idx, "validation":validation_idx, "test":test_idx}
        
        return partition, labels, label_index_ID_table

