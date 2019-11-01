
'''
Used to transcribe individual .wav sample into text.

'''

import os
import torch
from cnn_model import ConvNet2 as Net
import soundfile as wavfile
from data.front_end_processing import logfbank
from data.import_data import get_phoneme_index_dict
from torchvision.transforms import transforms
import argparse
from ctc_decoder import BeamSearchDecoder

parser = argparse.ArgumentParser(description="CSR Pytorch transcribe")
parser.add_argument('--path_to_wav', default="../data/GoogleSpeechCommands/wav_format/backward/0a2b400e_nohash_0.wav")
parser.add_argument('--path_to_model', default="./trained_models/checkpoint.pt")
parser.add_argument('--nfft', default=512)
parser.add_argument('--nfilt', default=70)
parser.add_argument('--window_size', default=0.02)
parser.add_argument('--step_size', default=0.01)
#parser.add_argument('--parse_', default=None)

args = parser.parse_args()


def load_model(model_path_in):
    model = Net()
    model.load_state_dict(torch.load(model_path_in))
    model.eval()
    return model


def wav_to_spec(wav_path_in):
    test_sound, samplerate = wavfile.read(wav_path_in)
    spec = logfbank(test_sound, samplerate,
                    winlen=args.window_size,
                    winstep=args.step_size,
                    nfilt=args.nfilt,
                    nfft=args.nfft)
    return spec


def evaluate_sample(spec, transform, model, decoder):
    input = transform(spec.T).float()
    input = input.unsqueeze(0)
    input_percentages = torch.FloatTensor(1)
    input_percentages[0] = 1.0
    input_lengths = input_percentages.mul_(int(input.shape[3])).int()
    output, output_lengths = model(input, input_lengths)
    decoded_sequence, scores, timesteps, out_seq_len = decoder.beam_search_batch(output, output_lengths)

    return decoded_sequence[0][0][0:out_seq_len[0][0]], scores[0][0], timesteps[0][0], out_seq_len[0][0]


model = load_model(args.path_to_model)
curDir = os.getcwd()
spec = wav_to_spec(args.path_to_wav)

decoder = BeamSearchDecoder()
transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.0],
                                  std=[0.5])])

decoded_sequence, scores, timesteps, out_seq_len = evaluate_sample(spec, transform, model, decoder)

_, index_phoneme_dict = get_phoneme_index_dict()
decoded_sequence = decoded_sequence.tolist()
translated_sequence = [index_phoneme_dict[x] for x in decoded_sequence]
print('Path: ', args.path_to_wav)
print('Predicted label sequence: ', translated_sequence)





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
        all_files = [x + '/' + y for y in os.listdir(dataset_path_in + x) if '.wav' in y]
        total += len(all_files)

        data_ID += all_files
        label_ID += ([label_index] * len(all_files))
        label_index += 1
    return data_ID, label_ID

#dataset_path = "../data/GoogleSpeechCommands/wav_format/"
#data_ID, label_ID = get_data_id(dataset_path)
