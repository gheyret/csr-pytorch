import math
import os

from data.data_importer import import_data_libri_speech


def get_unique_entires_list(input_list):
    output_list = set(input_list)
    output_list = list(output_list)
    return output_list


def get_all_words_dataset(dataset_path, vocabulary_path=None):
    # Remember to remove the break line in parse_trans_file_libri_speech
    list_id, label_dict, missing_words = import_data_libri_speech(dataset_path, vocabulary_path=vocabulary_path, verbose=False)
    missing_words = get_unique_entires_list(missing_words)  # Make sure they are all unique
    return missing_words


def get_all_words_ls(libri_speech_path, vocabulary_path=None):
    missing_words_all = []
    for x in os.listdir(libri_speech_path):
        x_path = os.path.join(libri_speech_path, x)
        if os.path.isdir(x_path):
            missing_words = get_all_words_dataset(x_path + "/", vocabulary_path)
            missing_words_all.extend(missing_words)
    missing_words_all = get_unique_entires_list(missing_words_all)
    return missing_words_all

def write_missing_words_to_file(missing_words, destination_file):
    with open(destination_file, "w") as outfile:
        for word in missing_words:
            outfile.write(word + "\n")

def split_missing_words_files(missing_words, destination_file, max_entries_per_file):
    num_words = len(missing_words)
    num_files = math.ceil(num_words/max_entries_per_file)
    destination_path = destination_file.strip('.txt')
    for file_i in range(num_files):
        min_idx_current_file = (file_i) * max_entries_per_file
        max_idx_current_file = (file_i + 1) * max_entries_per_file
        with open(destination_path + str(file_i) + ".txt", "w") as outfile:
            for i, word in enumerate(missing_words):
                if (i >= min_idx_current_file) & (i < max_idx_current_file):
                    outfile.write(word + "\n")
    return num_files

import argparse

parser = argparse.ArgumentParser(description="Get CMU dict")

parser.add_argument('--libri_path', default='../data/LibriSpeech/')
parser.add_argument('--destination_file', default='../data/LibriSpeech/missing_words.txt')
parser.add_argument('--vocabulary_path', default=None)  # If other external vocabulary is used, add the path to it here.
args = parser.parse_args()

missing_words = get_all_words_ls(args.libri_path, vocabulary_path=args.vocabulary_path)  # 90153 unique words.

# Here just add "get_all_words_xxx" for other datasets. Then extend the list with new missing words.
print(len(missing_words))
num_files = split_missing_words_files(missing_words, args.destination_file, 25000)
print("Saved the missing words into txt files at: " + args.destination_file)

print("Go to: http://www.speech.cs.cmu.edu/tools/lextool.html and add each of the files as word_file to get transcription")
print("use wget to download the .dict files.")
print("Rename the .dict files to 0.dict, 1.dict etc")

word_phoneme_dict = dict()
for file_idx in range(num_files):
    with open(args.libri_path + str(file_idx) + ".dict", 'r') as file:
        lines = file.readlines()
        for x in lines:
            x = x.replace("\t", " ")
            x = x.rstrip("\n")
            word_list = x.split(" ")
            word_phoneme_dict[word_list[0]] = word_list[1:]
with open(args.libri_path + "missing_words.dict", "w") as outfile:
    for key, value in word_phoneme_dict.items():
        line = ""
        if "(" not in key:  # remove duplicate pronounciations
            line = line + key.lower()
            for phoneme in value:
                if phoneme == "TS":
                    line = line + " EH S"
                else:
                    line = line + " " + phoneme
            line = line.replace("IX Z", "EH S")
            outfile.write(line+"\n")