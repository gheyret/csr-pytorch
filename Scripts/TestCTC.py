import torch
import torch.nn as nn
from csv_to_list import csvToList
import csv
T = 50      # Input sequence length
C = 20      # Number of classes (including blank)
N = 16      # Batch size
S = 30      # Target sequence length of longest target in batch
S_min = 10  # Minimum target length, for demonstration purposes

# Initialize random batch of input vectors, for *size = (T,N,C)
input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()

# Initialize random batch of targets (0 = blank, 1:C = classes)
target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)

input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
ctc_loss = nn.CTCLoss()
loss = ctc_loss(input, target, input_lengths, target_lengths)
loss.backward()

word_ID = 0

label_list = [' ', '_', 'AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'EHR', 'ER', 'EY',
                   'F',
                   'G', 'H', 'IH', 'IY', 'IYR', 'JH', 'K', 'L', 'M', 'N', 'NG', 'O', 'OW', 'OY', 'P', 'R', 'S',
                   'SH', 'T', 'TH', 'UH', 'UHR', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']

label_dict = dict()
for i, x in enumerate(label_list):
    label_dict[x] = i

#labels.sort()
print(label_list)
print(len(label_list))
output_list = []
path_to_csv = "./input_data_hdf5_70m/label_index_ID_table.csv"
with open(path_to_csv, newline='') as csvfile:
    output_list = list(csv.reader(csvfile))
output_list = [x for x in output_list if x]
output_dict = dict(output_list)

print(output_dict)
word_str = output_dict[str(word_ID)]
switcher = {
    0: "B AE K W AX D", #backward
    1: "B EH D",       #bed
    2: "B ER D",       #bird
    3: "K AE T",       #cat
    4: "D O G ",       #dog
    5: "D AW N",       #down
    6: "EY T",         #'eight',
    7: "F AY V",       #'five',
    8: "F O L OW",     #'follow',
    9: "F AO W AX D",  #'forward',
    10: "F AO",        #'four',
    11: "G OW",        #'go',
    12: "H AE P IY",   #'happy',
    13: "H AW S",      #'house',
    14: "L ER N",            #'learn',
    15: "L EH F T",            #'left',
    16: "M AA V IH N",            #'marvin',
    17: "N AY N",            #'nine',
    18: "N OW",            #'no',
    19: "O F",            #'off',
    20: "O N",            #'on',
    21: "W AH N",            #'one',
    22: "R AY T",            #'right',
    23: "S EH V AX N",            #'seven',
    24: "SH IY L AX",            #'sheila',
    25: "S IH K S",            #'six',
    26: "S T O P",            #'stop',
    27: "TH R IY",            #'three',
    28: "T R IY",            # 'tree',
    29: "T UW",            #'two',
    30: "AH P",            #'up',
    31: "V IH ZH UHR L",            #'visual',
    32: "W AW",             #'wow',
    33: "Y EH S",            #'yes',
    34: "Z IYR R OW"            #'zero'
}

phonetic_str = switcher.get(word_ID)
print(phonetic_str)
phonetic_list = phonetic_str.split(" ")
phonetic_id_list = [label_dict[x] for x in phonetic_list]

m = nn.Softmax(dim=-1)
input = torch.randn(2, 2, 1, 3)
output = m(input)
print(output)