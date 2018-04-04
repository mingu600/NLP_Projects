import torch
import torch.nn as nn
from torch.autograd import Variable

class Attention(nn.Module):
    def __init__(self, pad_token=1, bidirectional=True, h_dim=300, use_cuda = True):
        super(Attention, self).__init__()
        # Check attn type and store variables
        self.bidirectional = bidirectional
        self.h_dim = h_dim
        self.pad_token = pad_token

    def attention(self, in_e, out_e, out_d):
        '''Produces context and attention distribution'''

        # Deal with bidirectional encoder, move batches first
        if self.bidirectional: # sum hidden states for both directions
            out_e = out_e.contiguous().view(out_e.size(0), out_e.size(1), 2, -1).sum(2).view(out_e.size(0), out_e.size(1), -1)
        out_e = out_e.transpose(0,1) # b x sl x hd
        out_d = out_d.transpose(0,1) # b x tl x hd
        attn = out_e.bmm(out_d.transpose(1,2)) # (b x sl x hd) (b x hd x tl) --> (b x sl x tl)

        # Softmax and reshape
        attn = attn.exp() / attn.exp().sum(dim=1, keepdim=True) # in updated pytorch, make softmax
        attn = attn.transpose(1,2) # --> b x tl x sl

        # Get attention distribution
        context = attn.bmm(out_e) # --> b x tl x hd
        context = context.transpose(0,1) # --> tl x b x hd

        return context, attn

    def forward(self, in_e, out_e, out_d):
        '''Produces context using attention distribution'''
        context, attn = self.attention(in_e, out_e, out_d)
        return context

    def get_visualization(self, in_e, out_e, out_d):
        '''Gives attention distribution for visualization'''
        context, attn = self.attention(in_e, out_e, out_d)
        return attn
