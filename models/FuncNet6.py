import torch.nn as nn
import torch
from models.helper_functions import SequenceWise, ResBlock, PaddedRNN
import math


class FuncNet6(nn.Module):
    """Wav2Letter Speech Recognition model
        https://github.com/LearnedVector/Wav2Letter/blob/master/Wav2Letter/model.py
        Architecture is based off of Facebooks AI Research paper
        https://arxiv.org/pdf/1609.03193.pdf
        This specific architecture accepts mfcc or
        power spectrums speech signals
        TODO: use cuda if available
        Args:
            num_features (int): number of mfcc features
            num_classes (int): number of unique grapheme class labels
    """

    def __init__(self, num_classes = 46, num_features_input = 120, num_input_channels = 1, non_linearity = 'ReLU',
        memory_type = None, rnn_bidirectional = None, input_type = 'features'):
        super(FuncNet6, self).__init__()
        self.input_type = input_type

        if input_type is 'raw':
            self.layerRaw1 = nn.Sequential(
                nn.Conv1d(1, self.num_features_input, 251, stride=80, padding=125),  # 16
                nn.BatchNorm1d(self.num_features_input),
                self.non_linearity)
            self.layerRaw2 = nn.Sequential(
                nn.Conv1d(self.num_features_input, self.num_features_input, 49, stride=2, padding=24),  # 16*5
                nn.BatchNorm1d(self.num_features_input),
                self.non_linearity)

        # Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.layer1 = nn.Sequential(
            nn.Conv1d(num_features_input, 250, 49, 2, 24),
            nn.BatchNorm1d(250),
            torch.nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(250, 250, 7, padding=3),
            nn.BatchNorm1d(250),
            torch.nn.ReLU()
            )
        self.layer3 = nn.Sequential(
            nn.Conv1d(250, 250, 7, padding=3),
            nn.BatchNorm1d(250),
            torch.nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv1d(250, 250, 7, padding=3),
            nn.BatchNorm1d(250),
            torch.nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv1d(250, 250, 7, padding=3),
            nn.BatchNorm1d(250),
            torch.nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv1d(250, 250, 7, padding=3),
            nn.BatchNorm1d(250),
            torch.nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv1d(250, 250, 7, padding=3),
            nn.BatchNorm1d(250),
            torch.nn.ReLU())
        self.layer8 = nn.Sequential(
            nn.Conv1d(250, 250, 7, padding=3),
            nn.BatchNorm1d(250),
            torch.nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv1d(250, 2000, 33, padding=16),
            nn.BatchNorm1d(2000),
            torch.nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv1d(2000, 2000, 1),
            torch.nn.ReLU())
        self.layer11 = nn.Conv1d(2000, num_classes, 1)

        self.layers = nn.Sequential(
            nn.Conv1d(num_features_input, 250, 48, 2), # 1
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7), # 2
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7), # 3
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7), # 4
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7), # 5
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),# 6
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7), # 7
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7), # 8
            torch.nn.ReLU(),
            nn.Conv1d(250, 2000, 32), # 9
            torch.nn.ReLU(),
            nn.Conv1d(2000, 2000, 1), # 10
            torch.nn.ReLU(),
            nn.Conv1d(2000, num_classes, 1),
        )

        self.softMax = nn.LogSoftmax(dim=-1)
    def forward(self, x, input_lengths):
        """Forward pass through Wav2Letter network than
            takes log probability of output
        Args:
            batch (int): mini batch of data
             shape (batch, num_features, frame_len)
        Returns:
            log_probs (torch.Tensor):
                shape  (batch_size, num_classes, output_len)
        """
        N, _, F, T_in = x.size()
        if self.input_type is 'raw':
            x = x.view(N, F, T_in)
            x_input = self.layerRaw1(x)
            x_input = self.layerRaw2(x_input)
            x_input = x_input.unsqueeze(1)
        else:
            x_input = x.view(N, F, T_in)
        total_strides = 2
        if self.input_type is 'raw':
            total_strides = total_strides * 80 * 2
        time_refactoring = 1 / total_strides
        output_lengths = torch.ceil(input_lengths.float().mul_(time_refactoring)).int()
        # ((in - kw + 2 * Pad) / Stride) + 1)
        """
        output_lengths = torch.floor(torch.div((input_lengths.float() - 48 + 2 * 0), 2)) + 1
        output_lengths = torch.floor(torch.div((output_lengths - 7 + 2 * 0), 1)) + 1
        output_lengths = torch.floor(torch.div((output_lengths - 7 + 2 * 0), 1)) + 1
        output_lengths = torch.floor(torch.div((output_lengths - 7 + 2 * 0), 1)) + 1
        output_lengths = torch.floor(torch.div((output_lengths - 7 + 2 * 0), 1)) + 1
        output_lengths = torch.floor(torch.div((output_lengths - 7 + 2 * 0), 1)) + 1
        output_lengths = torch.floor(torch.div((output_lengths - 7 + 2 * 0), 1)) + 1
        output_lengths = torch.floor(torch.div((output_lengths - 7 + 2 * 0), 1)) + 1
        output_lengths = torch.floor(torch.div((output_lengths - 32 + 2 * 0), 1)) + 1
        output_lengths = (torch.floor(torch.div((output_lengths - 1 + 2 * 0), 1)) + 1).int()
        """

        # y_pred shape (batch_size, num_classes, output_len)
        out = self.layer1(x_input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        y_pred = self.layer11(out)


        #y_pred = self.layers(x_input)

        # compute log softmax probability on graphemes
        #log_probs = F.log_softmax(y_pred, dim=1)

        # NxCxT -> TxNxC for ctc_loss
        y_pred = y_pred.transpose(1, 2).transpose(0, 1)
        log_probs = self.softMax(y_pred)

        return log_probs, output_lengths
