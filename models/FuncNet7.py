import torch.nn as nn
import torch
from models.helper_functions import SequenceWise, ResBlock, PaddedRNN
import math


class FuncNet7(nn.Module):
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
        super(FuncNet7, self).__init__()
        self.input_type = input_type
        self.num_input_channels = num_input_channels
        if non_linearity is 'ReLU':
            self.non_linearity = nn.ReLU()
        elif non_linearity is 'Hardtanh':
            self.non_linearity = nn.Hardtanh(0, 20, inplace=True)

        if input_type is 'raw':
            self.layerRaw1 = nn.Sequential(
                nn.Conv1d(1, self.num_features_input, 251, stride=80, padding=125),  # 16
                nn.BatchNorm1d(self.num_features_input),
                self.non_linearity)
            self.layerRaw2 = nn.Sequential(
                nn.Conv1d(self.num_features_input, self.num_features_input, 49, stride=2, padding=24),  # 16*5
                nn.BatchNorm1d(self.num_features_input),
                self.non_linearity)

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.num_input_channels, 250, kernel_size=[5, 5], stride=(2, 2), padding=[2, 2]),
            torch.nn.ReLU(),
            nn.BatchNorm2d(250))
        self.layer2 = nn.Sequential(
            nn.Conv2d(250, 250, kernel_size=[5, 5], stride=(2,  2), padding=[2, 2]),
            nn.BatchNorm2d(250))
        self.layer3 = nn.Sequential(
            nn.Conv2d(250, 250, kernel_size=[3, 3], stride=(2, 1), padding=[1, 1]),  # 256
            nn.BatchNorm2d(250))
        self.layer3p = nn.Sequential(
            nn.Conv2d(250, 250, kernel_size=[3, 3], stride=(2, 1), padding=[1, 1]),  # 256
            nn.BatchNorm2d(250))
        self.layer3p2 = nn.Sequential(
            nn.Conv2d(250, 250, kernel_size=[3, 3], stride=(2, 1), padding=[1, 1]),  # 256
            nn.BatchNorm2d(250))
        self.layer3p3 = nn.Sequential(
            nn.Conv2d(250, 250, kernel_size=[3, 3], stride=(2, 1), padding=[1, 1]),  # 256
            nn.BatchNorm2d(250))
        self.layer3p4 = nn.Sequential(
            nn.Conv2d(250, 250, kernel_size=[3, 3], stride=(2, 1), padding=[1, 1]),  # 256
            nn.BatchNorm2d(250))
        self.layer4 = nn.Sequential(
            nn.Conv2d(250, 2000, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1]),  # 512
            nn.BatchNorm2d(2000))

        rnn_input_size = int(math.floor((num_features_input - 5 + 2 * 2) / 2) + 1)
        rnn_input_size = int(math.floor((rnn_input_size - 5 + 2 * 2) / 2) + 1)
        rnn_input_size = int(math.floor((rnn_input_size - 3 + 2 * 1) / 2) + 1)
        rnn_input_size = int(math.floor((rnn_input_size - 3 + 2 * 1) / 2) + 1)
        rnn_input_size = int(math.floor((rnn_input_size - 3 + 2 * 1) / 2) + 1)
        rnn_input_size = int(math.floor((rnn_input_size - 3 + 2 * 1) / 2) + 1)
        rnn_input_size = int(math.floor((rnn_input_size - 3 + 2 * 1) / 2) + 1)
        rnn_input_size = int(math.floor((rnn_input_size - 3 + 2 * 1) / 1) + 1)
        rnn_input_size = int(rnn_input_size * 2000)



        self.fc = nn.Sequential(
            nn.Linear(rnn_input_size, num_classes))

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
            x_input = x #x.view(N, F, T_in)
        total_strides = 2*2
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
        out = self.non_linearity(out)

        out = self.layer2(out)
        out = self.non_linearity(out)

        out = self.layer3(out)
        out = self.non_linearity(out)

        out = self.layer3p(out)
        out = self.non_linearity(out)

        out = self.layer3p2(out)
        out = self.non_linearity(out)

        out = self.layer3p3(out)
        out = self.non_linearity(out)

        out = self.layer3p4(out)
        out = self.non_linearity(out)

        out = self.layer4(out)
        out = self.non_linearity(out)

        N, FM, F, T_out = out.size()  # N = Batch size, FM = Feature maps, f = frequencies, t = time
        out = out.view(N, T_out, FM * F)

        y_pred = self.fc(out)

        #y_pred = self.layers(x_input)

        # compute log softmax probability on graphemes
        #log_probs = F.log_softmax(y_pred, dim=1)

        # NxTxC -> TxNxC for ctc_loss
        y_pred = y_pred.transpose(0, 1)
        log_probs = self.softMax(y_pred)

        return log_probs, output_lengths
