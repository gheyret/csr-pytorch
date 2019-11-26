import torch.nn as nn
import torch
from models.helper_functions import SequenceWise, ResBlock

class ConvNet5(nn.Module):
    """
    Same as ConvNet2 but using different kernel sizes at start and then concatinating the resulting filters.
    This allows different resolutions to be used when analyzing the spectrogram.
    """
    def __init__(self):
        super(ConvNet5, self).__init__()
        self.layer1x3 = nn.Sequential(
            nn.Conv2d(1, 22, kernel_size=[3, 3], stride=(2, 2), padding=[0, 1], dilation=1))  # 3x3

        self.layer1x5 = nn.Sequential(
            nn.Conv2d(1, 21, kernel_size=[5, 5], stride=(2, 2), padding=[1, 2], dilation=1))  # 5x5

        self.layer1x7 = nn.Sequential(
            nn.Conv2d(1, 21, kernel_size=[7, 7], stride=(2, 2), padding=[2, 3], dilation=1))  # 7x7
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Hardtanh(0, 20, inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=[5, 5], stride=(2,  2), padding=[0, 2]),
            nn.BatchNorm2d(128),
            nn.Hardtanh(0, 20, inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(2, 1), padding=[0, 1]),  # 256
            nn.BatchNorm2d(256),
            nn.Hardtanh(0, 20, inplace=True))
        self.layer3p = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1]),  # 256
            nn.BatchNorm2d(256),
            nn.Hardtanh(0, 20, inplace=True))
        self.layer3p2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1]),  # 256
            nn.BatchNorm2d(256),
            nn.Hardtanh(0, 20, inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(2, 1), padding=[0, 1]),  # 512
            nn.BatchNorm2d(512),
            nn.Hardtanh(0, 20, inplace=True))

        self.rnn = nn.Sequential(
            nn.LSTM(input_size=1536, hidden_size=512, num_layers=1, bidirectional=True, batch_first=False))
        self.batchNorm = SequenceWise(nn.BatchNorm1d(512))
        self.rnn2 = nn.Sequential(
            nn.LSTM(input_size=512, hidden_size=512, num_layers=1, bidirectional=True, batch_first=False))

        self.fc = nn.Sequential(
            nn.Linear(512, 46))
        self.softMax = nn.LogSoftmax(dim=-1)

    def forward(self, x, input_lengths):
        #Todo: Clean up
        N, _, F, T_in = x.size()
        out3x3 = self.layer1x3(x)
        out5x5 = self.layer1x5(x)
        out7x7 = self.layer1x7(x)
        out = torch.cat((out3x3, out5x5, out7x7), 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer3p(out)
        out = self.layer3p2(out)
        out = self.layer4(out)

        N, FM, F, T_out = out.size() # N = Batch size, FM = Feature maps, f = frequencies, t = time
        time_refactoring = 0.25 # T_out / T_in
        output_lengths = torch.ceil(input_lengths.float().mul_(time_refactoring)).int()
        out = out.view(N, FM*F, T_out)
        out = out.transpose(1, 2).transpose(0, 1).contiguous() # T x N x (F*FM)

        out = nn.utils.rnn.pack_padded_sequence(out, output_lengths, batch_first=False)
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