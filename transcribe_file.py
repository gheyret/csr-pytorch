
'''
Used to transcribe individual .wav sample into text.

'''

import os
import torch
from cnn_model import ConvNet
from scipy.io import wavfile
from data.front_end_processing import logfbank
from torchvision.transforms import transforms
from data.pytorch_dataloader_wav import GoogleSpeechEncoder



def get_data_id(dataset_path_in):
    sub_folder_list = []
    for x in os.listdir(dataset_path_in):
        if os.path.isdir(dataset_path_in + '/' + x):
            sub_folder_list.append(x)

    total = 0
    data_ID = []
    label_ID = []
    label_index = 0
    for x in (sub_folder_list[0:1]):  # (subFolderList[0:n_words])
        # get all the wave files
        all_files = [x + '/' + y for y in os.listdir(dataset_path + x) if '.wav' in y]
        total += len(all_files)

        data_ID += all_files
        label_ID += ([label_index] * len(all_files))
        label_index += 1
    return data_ID, label_ID

def load_model(model_path_in):
    model = ConvNet()
    model.load_state_dict(torch.load(model_path_in))
    model.eval()
    return model

def wav_to_spec(dataset_path_in, sample_ID):
    nfft = 512
    nfilt = 70
    window_size = 0.02  # s
    step_size = 0.01  # s
    samplerate, test_sound = wavfile.read(dataset_path_in + sample_ID)
    #test_sound = numpy.trim_zeros(test_sound, 'b')
    spec = logfbank(test_sound, samplerate,
                    winlen=window_size,
                    winstep=step_size,
                    nfilt=nfilt,
                    nfft=nfft)
    return spec

def decode_label(input_label):
    label_list = ['_', ' ', 'AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'EHR', 'ER',
                       'EY', 'F',
                       'G', 'H', 'IH', 'IY', 'IYR', 'JH', 'K', 'L', 'M', 'N', 'NG', 'O', 'OW', 'OY', 'P', 'R', 'S',
                       'SH', 'T', 'TH', 'UH', 'UHR', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']  # 0 = blank
    label_dict = dict()
    for i, x in enumerate(label_list):
        label_dict[i] = x
    label_dict = label_dict
    phoneme_list = [label_dict[int(x)] for x in input_label]
    return phoneme_list

def evaluate_sample(spec, transform, model):
    input = transform(spec.T).float()
    input = input.unsqueeze(0)
    output = model(input)
    return output


sample_idx = 5
dataset_path = "../data/GoogleSpeechCommands/wav_format/"
data_ID, label_ID = get_data_id(dataset_path)
# wav_path = dataset_path + "backward/0a24b400e_nohash_0"

# Load model
model_path = "./trained_models/CNN-BLSTMx2.pt"
model = load_model(model_path)

spec = wav_to_spec(dataset_path, data_ID[sample_idx])

transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.0],
                                  std=[0.5])])
output = evaluate_sample(spec, transform, model)
GSE = GoogleSpeechEncoder()
true_label = GSE.encode_labels(label_ID[sample_idx])
print(output)
print(true_label)
print(decode_label(true_label))

from ctc_decoder import greedy_decode_ctc, beam_ctc_decode

predicted_label_greedy = greedy_decode_ctc(output, balnk_code=0)
print(predicted_label_greedy)
phonetic_out = decode_label(predicted_label_greedy)
print(phonetic_out)
probs = output[:,0,:].detach().numpy()

labels, score = beam_ctc_decode(probs, balnk_code=0)
print(labels)
print(score)
# Predict and decode
#outputs = model(local_batch)

# 1. Load in all the names in the dataset so that we can iterate or choose freely
# 1.2 Convert label to CTC_label & compute length
# 2. Load the model
# 3. Use the model to get prob distribution
# 4. Send output to CTC decoder
# 5. Compute edit distance / PER