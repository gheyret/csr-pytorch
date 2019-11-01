import os
import csv



def get_phoneme_index_dict():
    label_list = ['_', ' ', 'AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AY', 'B',
                  'CH', 'D', 'DH', 'EH', 'EHR', 'ER', 'EY','F', 'G', 'H', 'IH',
                  'IY', 'IYR', 'JH', 'K', 'L', 'M', 'N', 'NG', 'O', 'OW',
                  'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UHR', 'UW',
                  'V', 'W', 'Y', 'Z', 'ZH']
    phoneme_index_dict = dict()
    index_phoneme_dict = dict()
    for i, x in enumerate(label_list):
        phoneme_index_dict[x] = i
        index_phoneme_dict[i] = x
    return phoneme_index_dict, index_phoneme_dict

def import_data_generated(dataset_path, verbose=False):
    '''

    :param dataset_path: dataset_path directs to the folder containing the wav samples
    :param verbose:
    :return: list_id and label_dict to be used by dataloader for this dataset. list_id is a list of filenames.
            label_dict is a dict where the keys are the file names and the lists are the labels in ints. eg
            label_dict["file_name_1.WAV"] = [3, 27, 33].
    '''


    # List all files in this directory. Store them in a list. This is list_id.
    # Take the last name in the folder, go to the parent dir and find name.txt.
    # Parse this file to find the file name and the corresponding phoneme list.

    list_id = os.listdir(dataset_path)
    dataset_path = dataset_path.rstrip("/")
    parent_dir = os.path.dirname(dataset_path)
    last_dir = os.path.basename(os.path.normpath(dataset_path))
    txt_file = parent_dir + "/" + last_dir + ".txt"

    list_id = []
    phoneme_index_dict, _ = get_phoneme_index_dict()

    label_dict = dict()
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        index = 0
        for x in lines:
            x = x.rstrip("\n")
            if "----" not in x:
                line_split = x.split(': ')

                category = line_split[0]
                value = line_split[1]

                if "Utterance" in category:
                    continue
                elif "FileName" in category:
                    file_name = value
                    list_id.append(file_name)
                elif "PhonemeList" in category:
                    label_phonemes = value.split(" ")
                    label_indexes = [phoneme_index_dict[y] for y in label_phonemes]
                    label_dict[file_name] = label_indexes

    return list_id, label_dict


def encode_labels_gsc(word_id, phoneme_index_dict):
    switcher = {
        "backward": "B AE K W AX D",  # backward
        "bed": "B EH D",  # bed
        "bird": "B ER D",  # bird
        "cat": "K AE T",  # cat
        "dog": "D O G",  # dog
        "down": "D AW N",  # down
        "eight": "EY T",  # 'eight',
        "five": "F AY V",  # 'five',
        "follow": "F O L OW",  # 'follow',
        "forward": "F AO W AX D",  # 'forward',
        "four": "F AO",  # 'four',
        "go": "G OW",  # 'go',
        "happy": "H AE P IY",  # 'happy',
        "house": "H AW S",  # 'house',
        "learn": "L ER N",  # 'learn',
        "left": "L EH F T",  # 'left',
        "marvin": "M AA V IH N",  # 'marvin',
        "nine": "N AY N",  # 'nine',
        "no": "N OW",  # 'no',
        "off": "O F",  # 'off',
        "on": "O N",  # 'on',
        "one": "W AH N",  # 'one',
        "right": "R AY T",  # 'right',
        "seven": "S EH V AX N",  # 'seven',
        "sheila": "SH IY L AX",  # 'sheila',
        "six": "S IH K S",  # 'six',
        "stop": "S T O P",  # 'stop',
        "three": "TH R IY",  # 'three',
        "tree": "T R IY",  # 'tree',
        "two": "T UW",  # 'two',
        "up": "AH P",  # 'up',
        "visual": "V IH ZH UHR L",  # 'visual',
        "wow": "W AW",  # 'wow',
        "yes": "Y EH S",  # 'yes',
        "zero": "Z IYR R OW"  # 'zero'
    }
    phonetic_str = switcher.get(word_id)
    # print(phonetic_str)
    phonetic_list = phonetic_str.split(" ")
    phonetic_id_list = [phoneme_index_dict[x] for x in phonetic_list]
    return phonetic_id_list


def import_data_gsc(dataset_path, verbose=False):
    '''

    :param dataset_path: Path to the GSC file structure containing the 3 txt files and folders with wav samples.
    :param verbose:
    :return:
    '''
    # Read the testing file and extract all IDs
    test_names = []
    f = open(dataset_path + "testing_list.txt", "r")
    if f.mode == 'r':
        contents = f.readlines()

    if verbose:
        print("number of test elements: %d" % len(contents))

    for s in contents:
        s = s.replace("\n", '')
        test_names.extend([s])

    # Read the validation file and extract all IDs
    validation_names = []
    f = open(dataset_path + "validation_list.txt", "r")
    if f.mode == 'r':
        contents = f.readlines()

    if verbose: print("number of validation elements: %d" % len(contents))

    for s in contents:
        s = s.replace("\n", '')
        validation_names.extend([s])

    # Extract ID of all files and store Label & name in categories train/test/validation
    sub_folder_list = []
    for x in os.listdir(dataset_path):
        if os.path.isdir(dataset_path + '/' + x):
            if "_background" not in x:
                sub_folder_list.append(x)

    test_label = dict()
    validation_label = dict()
    train_label = dict()

    train_names = []
    total = 0
    phoneme_index_dict, _ = get_phoneme_index_dict()

    for x in sub_folder_list:
        # get all the wave files
        all_files = [x + '/' + y for y in os.listdir(dataset_path + x) if '.wav' in y]
        total += len(all_files)
        label_list = encode_labels_gsc(x, phoneme_index_dict)
        for file_name in all_files:
            if file_name in test_names:
                test_label[file_name] = label_list
            elif file_name in validation_names:
                validation_label[file_name] = label_list
            else:
                train_names.append(file_name)
                train_label[file_name] = label_list

        # show file counts
        if verbose: print('count: %d : %s' % (len(all_files), x))
    if verbose: print("Total number of files: ", total)

    partition_names = {"train": train_names, "validation": validation_names, "test": test_names}
    partition_labels = {"train": train_label, "validation": validation_label, "test": test_label}

    return partition_names, partition_labels


def generate_csv_gsc(dataset_path, partition_names, partition_labels):

    for partition_type in ['test', 'validation', 'train']:
        if not os.path.exists(dataset_path + "list_id_" + partition_type):
            with open(dataset_path + "list_id_" + partition_type + ".csv", "w") as outfile:
                writer = csv.writer(outfile)
                partition = partition_names[partition_type]
                for file_name in partition:
                    writer.writerow([file_name])

        if not os.path.exists(dataset_path + "dict_labels_" + partition_type):
            with open(dataset_path + "dict_labels_" + partition_type + ".csv", "w") as outfile:
                writer = csv.writer(outfile)
                partition = partition_labels[partition_type]
                for key, val in partition.items():
                    writer.writerow([key, val])


def csv_to_list(path_to_csv):
    with open(path_to_csv) as csvfile:  # , newline='' in Windows
        output_list = list(csv.reader(csvfile))
        #output_list = numpy.asarray(output_list)[0]
        output_list = [x[0] for x in output_list]
        return output_list


def csv_to_dict(path_to_csv):
    dict_out = dict()
    with open(path_to_csv) as csvfile:  # , newline='' in Windows
        reader = csv.reader(csvfile)
        for key, value in reader:
            value = value.replace("[", "").replace("]", "")
            value = value.split(", ")
            value = list(map(int, value))
            dict_out[key] = value
        return dict_out

