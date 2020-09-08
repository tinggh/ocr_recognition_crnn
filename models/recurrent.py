import torch.nn as nn



class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nLayer, dropout, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, nLayer, dropout=dropout, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
            

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output



def DoubleBiLSTM(nIn, nOut, nHidden, nLayer, dropout):
    rnn = nn.Sequential(
            BidirectionalLSTM(nIn, nHidden, nLayer, dropout, nHidden),
            BidirectionalLSTM(nHidden, nHidden, nLayer, dropout, nOut)
            )
    return rnn

def singleBiLSTM(nIn, nOut, nHidden, nLayer, dropout):
    rnn = nn.Sequential(
            BidirectionalLSTM(nIn, nHidden, nLayer, dropout, nOut)
            )
    return rnn

def linear(nIn, nout, nHidden, nLayer, dropout):
    fc = nn.Linear(nIn, nout)
    return fc