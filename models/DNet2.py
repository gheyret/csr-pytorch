import torch.nn as nn
import torch
from models.helper_functions import SequenceWise, ResBlock
import math

class DNet2(nn.Module):
    """
    Baseline.
    LibriSpeech training on train-clean-100 and train-clean-360 and dev-clean as validation reaches ~20% PER on test-clean.
    """
    def __init__(self):
        super(DNet2, self).__init__()

        kernel_F_1 = 41
        kernel_T_1 = 11
        stride_F_1 = 2
        stride_T_1 = 2
        padding_F_1 = 20
        padding_T_1 = 5

        kernel_F_2 = 21
        kernel_T_2 = 11
        stride_F_2 = 2
        stride_T_2 = 1
        padding_F_2 = 10
        padding_T_2 = 5

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=[kernel_F_1, kernel_T_1], stride=(stride_F_1, stride_T_1),
                      padding=[padding_F_1, padding_T_1]),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=[kernel_F_2, kernel_T_2], stride=(stride_F_2, stride_T_2),
                      padding=[padding_F_2, padding_T_2]),
            nn.BatchNorm2d(32),
            nn.ReLU())

        rnn_input_size = int(math.floor((70 - kernel_F_1 + 2*padding_F_1)/stride_F_1) + 1)
        rnn_input_size = int(math.floor((rnn_input_size - kernel_F_2 + 2 * padding_F_2) / stride_F_2) + 1)
        rnn_input_size = int(rnn_input_size*32)

        self.rnn = nn.Sequential(
            nn.LSTM(input_size=rnn_input_size, hidden_size=768, num_layers=1, bidirectional=True, batch_first=False))
        self.batchNorm = SequenceWise(nn.BatchNorm1d(768))
        self.rnn2 = nn.Sequential(
            nn.LSTM(input_size=768, hidden_size=768, num_layers=1, bidirectional=True, batch_first=False))

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.Linear(768, 46, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )

        self.softMax = nn.LogSoftmax(dim=-1)

    def forward(self, x, input_lengths):
        N, _, F, T_in = x.size()
        out = self.layer1(x)
        out = self.layer2(out)

        N, FM, F, T_out = out.size() # N = Batch size, FM = Feature maps, f = frequencies, t = time
        time_refactoring = 0.5 # T_out / T_in
        output_lengths = torch.ceil(input_lengths.float().mul_(time_refactoring)).int()
        out = out.view(N, FM*F, T_out)
        out = out.transpose(1, 2).transpose(0, 1).contiguous() # T x N x (F*FM)

        out = nn.utils.rnn.pack_padded_sequence(out, output_lengths, batch_first=False)  # Handles varying length of input
        out, h = self.rnn(out)
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        out = out.view(out.size(0), out.size(1), 2, -1).sum(2).view(out.size(0), out.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum

        out = self.batchNorm(out)

        out = nn.utils.rnn.pack_padded_sequence(out, output_lengths, batch_first=False)
        out, h = self.rnn2(out)
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        out = out.view(out.size(0), out.size(1), 2, -1).sum(2).view(out.size(0), out.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum

        # FC: connect a tensor(N, *, in_features) to (N, *, out_features)
        out = self.fc(out) # TxNxC
        out = self.softMax(out)
        return out, output_lengths