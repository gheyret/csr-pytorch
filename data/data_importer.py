import os
import csv
from xml.dom import minidom



def concat_datasets(list_id_first, list_id_second, label_dict_first, label_dict_second, wav_path_first,
                    wav_path_second):
    '''
    Takes list_id and label_dict from two sources and concatenates them into one output for each.
    If this function has already been used, then the wav_path will be a dict. If that is the case, extend the dict.
    Otherwise create a new dict.
    '''
    if type(wav_path_first) is not dict:
        wav_path = dict()
        for id in list_id_first:
            wav_path[id] = wav_path_first
    else:
        wav_path = wav_path_first.copy()
    for id in list_id_second:
        wav_path[id] = wav_path_second


    list_id = list_id_first.copy()
    list_id.extend(list_id_second)
    label_dict = label_dict_first.copy()
    label_dict.update(label_dict_second)

    return list_id, label_dict, wav_path


def get_num_classes(label_type='phoneme'):
    label_index_dict, index_label_dict = get_label_index_dict(label_type)
    return len(label_index_dict)


def get_label_index_dict(label_type='phoneme'):
    if label_type is 'phoneme':
        label_list_eng = ['_', '-', 'AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AY', 'B',
                          'CH', 'D', 'DH', 'EH', 'EHR', 'ER', 'EY', 'F', 'G', 'H', 'IH',
                          'IY', 'IYR', 'JH', 'K', 'L', 'M', 'N', 'NG', 'O', 'OW',
                          'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UHR', 'UW',
                          'V', 'W', 'Y', 'Z', 'ZH']  # - is space character
        label_list_cmu = ['_', '-', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B',
                          'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH',
                          'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW',
                          'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW',
                          'V', 'W', 'Y', 'Z', 'ZH']
        label_list = label_list_cmu
    elif label_type is 'letter':
        label_list = ['_', ' ', '\'', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                      'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'Z']  # - is space character
    label_index_dict = dict()
    index_label_dict = dict()
    for i, x in enumerate(label_list):
        label_index_dict[x] = i
        index_label_dict[i] = x
    return label_index_dict, index_label_dict


def get_word_phoneme_dictionary(vocabulary_path, xml=True):
    if xml:
        eng_ipa_vocabulary = minidom.parse(vocabulary_path)
        term_nodes = eng_ipa_vocabulary.getElementsByTagName("DictionaryTerm")
        word_nodes = eng_ipa_vocabulary.getElementsByTagName("Word")

        word_phoneme = dict()
        for i, word_node, in enumerate(word_nodes):
            word = word_node.childNodes[0].data

            attribute_nodes = term_nodes[i].childNodes
            phoneme_nodes = attribute_nodes[4].childNodes
            phoneme_list = []
            for phoneme_node in phoneme_nodes:
                phoneme_str = phoneme_node.childNodes[0].data
                phoneme_list.append(phoneme_str)
            word_phoneme[word] = phoneme_list
        return word_phoneme
    else:
        word_phoneme = dict()
        with open(vocabulary_path, 'r') as file:
            lines = file.readlines()
            for x in lines:
                x = x.rstrip("\n")
                word_list = x.split(" ")
                word_phoneme[word_list[0]] = word_list[1:]
        return word_phoneme




def csv_to_list(path_to_csv):
    with open(path_to_csv) as csvfile:  # , newline='' in Windows
        output_list = list(csv.reader(csvfile))
        # output_list = numpy.asarray(output_list)[0]
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


def check_similiar_words(word, word_phoneme_dict, verbose):
    '''
    if word.endswith("less"):
        test_word = word[:-4]
        if test_word in word_phoneme_dict:
            if verbose:
                print(test_word + " exist, using that instead ++++++++++")
            phoneme_list_word = word_phoneme_dict[test_word].copy()
            phoneme_list_word.extend(["L", "EH", "S"])
            return True, phoneme_list_word

    if word.endswith("ous"):
        test_word = word[:-3]
        if test_word in word_phoneme_dict:
            if verbose:
                print(test_word + " exist, using that instead ++++++++++")
            phoneme_list_word = word_phoneme_dict[test_word].copy()
            phoneme_list_word.extend(["R", "AX", "S"])
            return True, phoneme_list_word
    '''
    # If word ends with "ed" can try to remove this and append the phoneme for "ed" - "D" (smile/smiled)
    if word.endswith("ed"):
        test_word = word[:-2]
        if test_word in word_phoneme_dict:
            if verbose:
                print(test_word + " exist, using that instead ++++++++++")
            phoneme_list_word = word_phoneme_dict[test_word].copy()
            phoneme_list_word.append("D")
            return True, phoneme_list_word

    if word.endswith("d"):
        test_word = word[:-1]
        if test_word in word_phoneme_dict:
            if verbose:
                print(test_word + " exist, using that instead ++++++++++")
            phoneme_list_word = word_phoneme_dict[test_word].copy()
            phoneme_list_word.append("D")
            return True, phoneme_list_word

    # If word ends with "s" can try to remove this and append the phoneme for "s" - "Z" (smith/smiths)
    if word.endswith("s"):
        test_word = word[:-1]
        if test_word in word_phoneme_dict:
            if verbose:
                print(test_word + " exist, using that instead ++++++++++")
            phoneme_list_word = word_phoneme_dict[test_word].copy()
            phoneme_list_word.append("S")
            return True, phoneme_list_word

    return False, []





def randomly_partition_data(part_size, list_id, label_dict):
    import numpy
    numpy.random.seed(123)
    num_samples = len(list_id)
    indices = numpy.random.permutation(num_samples)

    split_idx = int(numpy.round(num_samples * part_size))
    first_idx, second_idx = indices[:split_idx], indices[split_idx:]
    list_id_first = [list_id[idx] for idx in first_idx]
    list_id_second = [list_id[idx] for idx in second_idx]

    key_list_first = [list(label_dict)[idx] for idx in first_idx]
    label_dict_first = {key: label_dict[key] for key in key_list_first}

    key_list_second = [list(label_dict)[idx] for idx in second_idx]
    label_dict_second = {key: label_dict[key] for key in key_list_second}

    return list_id_first, list_id_second, label_dict_first, label_dict_second


def order_data_by_length(label_dict):
    def sort_by(p):
        return len(p[1])

    sorted_list_id = []
    sorted_label_dict = dict()
    sorted_label_tuples = sorted(label_dict.items(), key=sort_by)
    for file_label_tuple in sorted_label_tuples:
        file_name = file_label_tuple[0]
        label = file_label_tuple[1]
        sorted_list_id.append(file_name)
        sorted_label_dict[file_name] = label
    return sorted_list_id, sorted_label_dict


def print_label_distribution(label_dict):
    import numpy
    label_counter = numpy.zeros((2, 46))
    total = 0
    for i in range(0, 46):
        label_counter[0, i] = i
    for key, value in label_dict.items():
        for phoneme in value:
            label_counter[1, phoneme] += 1
            total += 1
    label_counter[1, :] = label_counter[1, :] / total

    _, index_phoneme_dict = get_label_index_dict()
    lines = []
    row = label_counter[0, :]
    lines.append('   '.join('{1:>{0}}'.format(4, index_phoneme_dict[int(x)]) for x in row))
    row = label_counter[1, :]
    lines.append(' '.join('{:.4f}'.format(x) for x in row))
    print('\n'.join(lines))


def import_data_generated(dataset_path):
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
    phoneme_index_dict, _ = get_label_index_dict()

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


def import_data_gsc(dataset_path, verbose=False, train_data_partition_size=0.0):
    '''

    :param train_data_partition_size:
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
    phoneme_index_dict, _ = get_label_index_dict()

    import random

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
                r = random.uniform(0, 1)
                if r > train_data_partition_size:
                    train_names.append(file_name)
                    train_label[file_name] = label_list

        # show file counts
        if verbose: print('count: %d : %s' % (len(all_files), x))
    if verbose: print("Total number of files: ", total)

    partition_names = {"train": train_names, "validation": validation_names, "test": test_names}
    partition_labels = {"train": train_label, "validation": validation_label, "test": test_label}

    return partition_names, partition_labels


def parse_trans_file_libri_speech(txt_file, sub_folder, missing_words, list_id, label_dict, label_index_dict,
                                  label_type, word_phoneme_dict=None, verbose=False):
    '''
    if label_type is 'phoneme' and word_phoneme_dict is empty. Then no translation should be given, and all words should
    be returned as missing.

    '''
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        for x in lines:
            # x = x.replace("'", "")
            x = x.rstrip("\n")
            word_list = x.split(" ")

            found_ipa_translation = True
            label_list = []
            file_name = sub_folder + word_list[0] + ".flac"
            if label_type is 'phoneme':
                for word in word_list[1:]:
                    word = word.lower()
                    if word in word_phoneme_dict:
                        phoneme_list_word = word_phoneme_dict[word]
                    else:
                        if verbose:
                            print(word + " doesn't exist in vocabulary ----------")
                        found_ipa_translation, phoneme_list_word = check_similiar_words(word, word_phoneme_dict,
                                                                                        verbose)
                    if not found_ipa_translation:
                        missing_words.append(word)
                        # break  # If break is set here, not all missing words will be found.
                    else:
                        if label_list:
                            label_list.append("-")
                        label_list.extend(phoneme_list_word)

                if found_ipa_translation:
                    list_id.append(file_name)
                    phoneme_list_idx = []
                    for phoneme in label_list:
                        phoneme_list_idx.append(label_index_dict[phoneme])
                    label_dict[file_name] = phoneme_list_idx
                    if verbose:
                        print("+++ " + x + " ...was added to the list")
                        print(phoneme_list_idx)
                else:
                    if verbose:
                        print("--- " + x + " ...wasn't added to the list")

            elif label_type is 'letter':
                for word in word_list[1:]:
                    letter_list_word_idx = []
                    for letter in word:
                        letter_list_word_idx.append(label_index_dict[letter])
                    if label_list:
                        label_list.append(1)  # index 1 in the label list should contain the space character.
                    label_list.extend(letter_list_word_idx)
                label_dict[file_name] = label_list
                list_id.append(file_name)
                if verbose:
                    print("Extracted: " + label_list)
                    print("--- " + x + " | was added to the list")


def import_data_libri_speech(dataset_path, vocabulary_path=None, vocabulary_path_is_xml=True,
                             vocabulary_path_addition=None, vocabulary_path_addition_is_xml=False,
                             label_type='phoneme', verbose=False):
    """

    """
    sub_folder_list = []
    trans_txt_list = []
    for x in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, x)):
            for y in os.listdir(dataset_path + '/' + x):
                if os.path.isdir(dataset_path + '/' + x + "/" + y):
                    sub_folder_list.append(x + "/" + y + "/")
                    trans_txt_list.append(x + "-" + y + ".trans.txt")

    word_phoneme_dict = dict()
    if (label_type is 'phoneme') & (vocabulary_path is not None):
        word_phoneme_dict = get_word_phoneme_dictionary(vocabulary_path, xml=vocabulary_path_is_xml)
        if vocabulary_path_addition is not None:
            word_phoneme_dict2 = get_word_phoneme_dictionary(vocabulary_path_addition,
                                                             xml=vocabulary_path_addition_is_xml)
            word_phoneme_dict.update(word_phoneme_dict2)
    label_index_dict, _ = get_label_index_dict(label_type=label_type)

    missing_words = []
    list_id = []
    label_dict = dict()
    for i, sub_folder in enumerate(sub_folder_list):
        txt_file = dataset_path + sub_folder_list[i] + trans_txt_list[i]
        if label_type is 'phoneme':
            parse_trans_file_libri_speech(txt_file, sub_folder, missing_words, list_id, label_dict, label_index_dict,
                                          label_type, word_phoneme_dict=word_phoneme_dict, verbose=verbose)
        else:
            parse_trans_file_libri_speech(txt_file, sub_folder, missing_words, list_id, label_dict, label_index_dict,
                                          label_type, word_phoneme_dict=None, verbose=verbose)
    if verbose:
        print(len(missing_words))
        print(len(list_id))

    return list_id, label_dict, missing_words

# dataset_path = "/media/olof/SSD 1TB/data/LibriSpeech/LibriSpeech/dev-clean/"  # 84/121123"
# vocabulary_path = "/media/olof/SSD 1TB/data/BritishEnglish_Reduced.xml"

# list_id, label_dict, missing_words = import_data_libri_speech(dataset_path, vocabulary_path)
# order_data_by_length(list_id, label_dict)
# randomly_partition_data(0.8, list_id, label_dict)
