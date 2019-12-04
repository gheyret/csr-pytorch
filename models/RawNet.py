import torch.nn as nn
import torch
from models.helper_functions import SequenceWise, ResBlock

class RawNet(nn.Module):
    """
    Baseline.
    LibriSpeech training on train-clean-100 and train-clean-360 and dev-clean as validation reaches ~20% PER on test-clean.
    Takes 70 mel-scale features as input.
    """
    def __init__(self):
        super(RawNet, self).__init__()
        self.n_features = 250
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, self.n_features, 251, stride=160, padding=125),  # 16
            nn.BatchNorm1d(self.n_features),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(self.n_features, self.n_features, 49, stride=2, padding=24),  # 16*5
            nn.BatchNorm1d(self.n_features),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv1d(250, 500, 5, stride=1, padding=2),  # 16*5*4 = 32*10
            nn.BatchNorm1d(500),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv1d(500, 500, 5, stride=1, padding=2),  # 320*4 -> 0.25 time factor
            nn.BatchNorm1d(500),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv1d(500, 1000, 5, stride=1, padding=2),  # 320*4 -> 0.25 time factor
            nn.BatchNorm1d(1000),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv1d(1000, 1500, 5, stride=1, padding=2),  # 320*4 -> 0.25 time factor
            nn.BatchNorm1d(1500),
            nn.ReLU())

        self.rnn = nn.Sequential(
            nn.LSTM(input_size=1500, hidden_size=500, num_layers=1, bidirectional=True, batch_first=False))
        self.batchNorm = SequenceWise(nn.BatchNorm1d(500))
        self.rnn2 = nn.Sequential(
            nn.LSTM(input_size=500, hidden_size=500, num_layers=1, bidirectional=True, batch_first=False))

        self.fc = nn.Sequential(
            nn.Linear(500, 46))
        self.softMax = nn.LogSoftmax(dim=-1)

    def forward(self, x, input_lengths):
        N, _, F, T_in = x.size()
        x = x.view(N, F, T_in)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)


        N, FM, T_out = out.size() # N = Batch size, FM = Feature maps, f = frequencies, t = time
        time_refactoring = 1/(160*2)  # T_out / T_in
        output_lengths = torch.ceil(input_lengths.float().mul_(time_refactoring)).int()
        out = out.view(N, FM, T_out)
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