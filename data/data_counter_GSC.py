from data.import_data_GSC import import_data
import os
import csv
from os.path import dirname


dataset_path_relative = "/data/GoogleSpeechCommands/wav_format/"

parent_dir = dirname(os.getcwd())
dataset_path = parent_dir + dataset_path_relative
if not os.path.exists(dataset_path):
    grandparent_dir = dirname(dirname(os.getcwd()))
    dataset_path = grandparent_dir + dataset_path_relative

dataset_path = "D:\data\GoogleSpeechCommands\wav_format/"

partition, labels, label_index_ID_table = import_data(dataset_path)

training_dict = dict()
word_counter_dict = dict()
csv_header = [""]
total = [0]*len(partition.keys())
for i, key in enumerate(partition.keys()): # key = train, test, validation
    csv_header.append(str(key))
    current_dict = dict()
    for j,  item in enumerate(partition[key]):
        split_list = item.split("/")
        word = split_list[0]

        if word not in word_counter_dict.keys():
            word_counter_dict[word] = [0]*len(partition.keys())
        word_counter_dict[word][i] += 1
        total[i] += 1
    #word_counter_dict[key] = current_dict

count_summary_csv_path = dataset_path + "count_summary"
if not os.path.exists(count_summary_csv_path):
    with open(count_summary_csv_path + ".csv", "w", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(csv_header)
        for key in word_counter_dict.keys():
            new_row = [key, *word_counter_dict[key]]
            writer.writerow(new_row)
        writer.writerow(["", *total])



