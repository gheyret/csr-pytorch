from data.import_data import import_data_gsc
import os
import csv
from os.path import dirname
import argparse

parser = argparse.ArgumentParser(description="CSR Pytorch")
parser.add_argument('--dataset_path_relative', default="/data/GoogleSpeechCommands/wav_format/")
parser.add_argument('--dataset_path', default=None)

args = parser.parse_args()
if args.dataset_path is None:
    parent_dir = dirname(os.getcwd())
    dataset_path = parent_dir + args.dataset_path_relative
    if not os.path.exists(dataset_path):
        grandparent_dir = dirname(dirname(os.getcwd()))
        dataset_path = grandparent_dir + args.dataset_path_relative
else:
    dataset_path = args.dataset_path

partition, labels = import_data_gsc(dataset_path, verbose=True)

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



