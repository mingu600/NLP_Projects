import torch
import torch.nn as nn
from torch.autograd import Variable

class DecoderLSTM(nn.Module):
    def __init__(self, embedding, num_layers = 2, h_dim_out, dropout_p=0.25, bidirectional = True, use_cuda = True):
        super(DecoderLSTM, self).__init__()
        self.vocab_size, self.embedding_size = embedding.size()
        self.num_layers = num_layers * 2 if bidirectional else self.num_layers = num_layers
        self.h_dim_out = h_dim_out
        self.dropout = dropout

        # Create word embedding, LSTM
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding.weight.data.copy_(embedding)
        self.lstm = nn.LSTM(self.embedding_size, self.h_dim_out, self.num_layers, dropout=self.dropout)
        self.dropout_l = nn.Dropout(self.dropout)

    def forward(self, input_seq, hidden_0):

        # Embed text and pass through GRU
        emb = self.embedding(input_seq)
        emb = self.dropout_l(emb)
        output, hidden = self.lstm(emb, hidden_0)
        return output, hidden
