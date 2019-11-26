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
