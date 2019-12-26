import torch.nn as nn
import torch
from models.helper_functions import SequenceWise, ResBlock, PaddedRNN, stack_frames
import math



class FuncNet8(nn.Module):
    """
    Adaptive layout.

    """
    def __init__(self, num_classes=46, num_features_input=70, num_input_channels=1, non_linearity='ReLU',
                 memory_type='GRU', rnn_bidirectional=False, input_type='features'):
        super(FuncNet8, self).__init__()
        self.input_type = input_type
        self.num_features_input = num_features_input
        self.num_input_channels = num_input_channels
        self.stacked_frames = 3
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
        rnn_hidden_size = 512
        self.rnn1 = PaddedRNN(mode=memory_type, input_size=self.num_features_input*self.stacked_frames,
                              hidden_size=rnn_hidden_size, num_layers=1, bidirectional=rnn_bidirectional,
                              batchnorm=True, sum_bidirectional=False)
        self.rnn2 = PaddedRNN(mode=memory_type, input_size=rnn_hidden_size*2,
                              hidden_size=rnn_hidden_size, num_layers=1, bidirectional=rnn_bidirectional, batchnorm=True,
                              sum_bidirectional=False)
        self.rnn3 = PaddedRNN(mode=memory_type, input_size=rnn_hidden_size * 2,
                              hidden_size=rnn_hidden_size, num_layers=1, bidirectional=rnn_bidirectional,
                              batchnorm=True,
                              sum_bidirectional=False)
        self.rnn4 = PaddedRNN(mode=memory_type, input_size=rnn_hidden_size * 2,
                              hidden_size=rnn_hidden_size, num_layers=1, bidirectional=rnn_bidirectional,
                              batchnorm=True,
                              sum_bidirectional=False)
        self.rnn5 = PaddedRNN(mode=memory_type, input_size=rnn_hidden_size * 2,
                              hidden_size=rnn_hidden_size, num_layers=1, bidirectional=rnn_bidirectional,
                              batchnorm=True,
                              sum_bidirectional=False)

        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden_size*2, num_classes))
        self.softMax = nn.LogSoftmax(dim=-1)

    def forward(self, x, input_lengths):
        N, _, F, T_in = x.size()
        if self.input_type is 'raw':
            x = x.view(N, F, T_in)
            x_input = self.layerRaw1(x)
            x_input = self.layerRaw2(x_input)
            x_input = x_input.unsqueeze(1)
        else:
            x_input = x
        N, FM, F, T_out = x_input.size() # N = Batch size, FM = Feature maps, f = frequencies, t = time
        total_strides = 1
        if self.input_type is 'raw':
            total_strides = total_strides*80*2
        time_refactoring = 1/total_strides  # 0.25 # T_out / T_in
        output_lengths = torch.ceil(input_lengths.float().mul_(time_refactoring)).int()

        out = x_input.view(N, FM*F, T_out)
        out = out.transpose(1, 2).transpose(0, 1).contiguous() # T x N x (F*FM)
        out, output_lengths = stack_frames(out, output_lengths, self.stacked_frames)  # Stack 3 frames and feed every 3rd
        out = self.rnn1(out, output_lengths)
        out = self.rnn2(out, output_lengths)
        out = self.rnn3(out, output_lengths)
        out = self.rnn4(out, output_lengths)
        out = self.rnn5(out, output_lengths)

        # FC: connect a tensor(N, *, in_features) to (N, *, out_features)
        out = self.fc(out) # TxNxC
        out = self.softMax(out)
        return out, output_lengths