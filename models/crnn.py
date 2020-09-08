import models.convnet as ConvNets
import models.recurrent as SeqNets
import torch.nn as nn
import torch.nn.parallel
import torch

class CRNN(nn.Module):
    def __init__(self, CNN, RNN, nIn, n_class, nHidden, nLayer, dropout):
        super(CRNN, self).__init__()
        self.cnn = ConvNets.__dict__[CNN]()
        print('Constructing {}'.format(RNN))
        self.rnn = SeqNets.__dict__[RNN](nIn, n_class, nHidden, nLayer, dropout)

    def forward(self, input):
        c_feat = self.cnn(input)

        b, c, h, w = c_feat.size()
        assert h == 1, "the height of the conv must be 1"

        c_feat = c_feat.squeeze(2)
        c_feat = c_feat.permute(2, 0, 1)

        output = self.rnn(c_feat)
        return output



