import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

class EncoderLSTM(nn.Module):
    def __init__(self, embedding, h_dim_out, num_layers= 2, dropout=0.25, bidirectional=True, use_cuda = True):
        super(EncoderLSTM, self).__init__()
        self.vocab_size, self.hidden_size = embedding.size()
        self.num_layers = num_layers
        self.h_dim_out = h_dim
        self.dropout = dropout
        self.bidirectional = bidirectional

        # Create word embedding and LSTM
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embedding.weight.data.copy_(embedding)
        self.lstm = nn.LSTM(self.hidden_size, self.h_dim_out, self.num_layers, dropout=self.dropout, bidirectional=bidirectional)
        self.dropout_l = nn.Dropout(dropout)
        self.use_cuda = use_cuda

    def forward(self, input):

        # Embed text
        emb = self.embedding(input)
        emb = self.dropout_l(emb)

        # Create initial hidden state of zeros: 2-tuple of num_layers x batch size x hidden dim
        num_layers = self.num_layers * 2 if self.bidirectional else self.num_layers
        init_h = Variable(torch.zeros(num_layers, input.size(1), self.h_dim), requires_grad=False)
        if self.use_cuda:
            init_h = init_h.cuda()
        hidden_0 = (init_h, init_h.clone())

        # Pass through LSTM
        output, hidden = self.lstm(emb, hidden_0)
        return output, hidden
