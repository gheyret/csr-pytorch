import torch.nn as nn
import torch


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        By SeanNaren/Deepspeech
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class ResBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        #self.bn0 = nn.BatchNorm2d(num_filters)
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1], dilation=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1], dilation=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class PaddedRNN(nn.Module):

    def __init__(self, mode='RNN', input_size=512, hidden_size=512, num_layers=1, bidirectional=True, batchnorm=True,
                 sum_bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        if mode is 'RNN':
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=False)
        elif mode is 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               bidirectional=bidirectional, batch_first=False)
        elif mode is 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               bidirectional=bidirectional, batch_first=False)
        self.use_batchnorm = batchnorm
        self.sum_bidirectional = sum_bidirectional
        if batchnorm:
            if sum_bidirectional:
                self.batchNorm = SequenceWise(nn.BatchNorm1d(hidden_size))
            else:
                self.batchNorm = SequenceWise(nn.BatchNorm1d(hidden_size*2))

    def forward(self, x, input_lengths):
        # Expects: T x N x (F*FM)
        out = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=False)
        out, _ = self.rnn(out)
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        if self.bidirectional & self.sum_bidirectional:
            out = out.view(out.size(0), out.size(1), 2, -1).sum(2).view(out.size(0), out.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum

        if self.use_batchnorm:
            out = self.batchNorm(out)
        return out


def stack_frames(x, lengths, num_stack):
    # T x N x F input
    # output: T/3 x N x F*3
    import math
    T, N, F = x.shape
    T_out = math.ceil(T / num_stack)


    x_stacked = torch.zeros([T_out, N, F * num_stack]).cuda()
    count = 0
    for i in range(math.floor(T / num_stack)):
        t = i * num_stack
        y = torch.cat([x[t+j, :, :] for j in range(num_stack)], dim=1)
        x_stacked[count, :, :] = y
        count = count + 1
    i = i + 1
    T_left = x.shape[0] - (i * num_stack)
    lengths_stacked = torch.floor(torch.div(lengths.float(), num_stack)).int()
    #lengths_stacked = lengths_stacked + T_left
    if T_left > 0:
        for j in range(T_left):
            t = j * num_stack
            index = i * num_stack + j
            x_stacked[count, :, t:(t + F)] = x[index, :, :]
    return x_stacked, lengths_stacked
