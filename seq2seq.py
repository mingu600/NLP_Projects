import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

from .Encoder import EncoderLSTM
from .Decoder import DecoderLSTM
from .Attention import Attention

class Seq2seq(nn.Module):
    def __init__(self, embedding_src, embedding_trg, h_dim, num_layers=2, dropout, bidirectional=True, attn_type, tokens_bos_eos_pad_unk=[0,1,2,3], use_cuda = True):
        super(Seq2seq, self).__init__()
        # Store hyperparameters
        self.h_dim = h_dim
        self.vocab_size_trg, self.emb_dim_trg = embedding_trg.size()
        self.bos_token = tokens_bos_eos_pad_unk[0]
        self.eos_token = tokens_bos_eos_pad_unk[1]
        self.pad_token = tokens_bos_eos_pad_unk[2]
        self.unk_token = tokens_bos_eos_pad_unk[3]
        # Create encoder, decoder, attention
        self.encoder = EncoderLSTM(embedding_src, h_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, use_cuda = use_cuda)
        self.decoder = DecoderLSTM(embedding_trg, h_dim, num_layers = num_layers, dropout=dropout, bidirectional = bidirectional, use_cuda = use_cuda)
        self.attention = Attention(pad_token=self.pad_token, bidirectional=bidirectional, h_dim=h_dim)
        # Create linear layers to combine context and hidden state
        self.l1 = nn.Linear(2 * self.h_dim, self.emb_dim_trg)
        self.tanh = nn.Tanh()
        self.dropout_l = nn.Dropout(dropout)
        self.l2 = nn.Linear(self.emb_dim_trg, self.vocab_size_trg)
        self.use_cuda = use_cuda

    def forward(self, input_seqs, target_seqs):
        if self.use_cuda:
            input_seqs = input_seqs.cuda()
        # Encode
        encoder_output, hidden_e = self.encoder(input_seqs)
        # Decode
        decoder_output, hidden_d = self.decoder(target_seqs, hidden_e)
        # Attend
        context = self.attention(input_seqs, encoder_output, decoer_output)
        output = torch.cat((decoder_output, context), dim=2)
        # Predict (returns probabilities)
        prediction = self.l1(output)
        prediction = self.dropout_l(self.tanh(prediction))
        prediction = self.l2(prediction)
        return prediction

    def predict_best(self, input_seqs):
        return self.beam_search(src, 1, max_len=30)[0][1]

    def predict_k(self, src, k, max_len=30, remove_tokens=[]):
        '''Predict top k possibilities for first max_len words.'''
        beam_outputs = self.beam_search(src, k, max_len=max_len, remove_tokens=remove_tokens) # returns top k options (as list of tuples)
        topk = [option[1] for option in beam_outputs] # list of k lists of word indices (as ints)
        return topk

    def beam_search(self, src, beam_size, max_len, remove_tokens=[]):
        '''Returns top beam_size sentences using beam search. Works only when src has batch size 1.'''
        if use_gpu: src = src.cuda()
        # Encode
        outputs_e, states = self.encoder(src) # batch size = 1
        # Start with '<s>'
        init_lprob = -1e10
        init_sent = [self.bos_token]
        best_options = [(init_lprob, init_sent, states)] # beam
        # Beam search
        k = beam_size # store best k options
        for length in range(max_len): # maximum target length
            options = [] # candidates
            for lprob, sentence, current_state in best_options:
                # Prepare last word
                last_word = sentence[-1]
                if last_word != self.eos_token:
                    last_word_input = Variable(torch.LongTensor([last_word]), volatile=True).view(1,1)
                    if use_gpu: last_word_input = last_word_input.cuda()
                    # Decode
                    outputs_d, new_state = self.decoder(last_word_input, current_state)
                    # Attend
                    context = self.attention(src, outputs_e, outputs_d)
                    out_cat = torch.cat((outputs_d, context), dim=2)
                    x = self.l1(out_cat)
                    x = self.dropout(self.tanh(x))
                    x = self.l2(x)
                    x = x.squeeze().data.clone()
                    # Block predictions of tokens in remove_tokens
                    for t in remove_tokens: x[t] = -10e10
                    lprobs = torch.log(x.exp() / x.exp().sum()) # log softmax
                    # Add top k candidates to options list for next word
                    for index in torch.topk(lprobs, k)[1]:
                        option = (float(lprobs[index]) + lprob, sentence + [index], new_state)
                        options.append(option)
                else: # keep sentences ending in '</s>' as candidates
                    options.append((lprob, sentence, current_state))
            options.sort(key = lambda x: x[0], reverse=True) # sort by lprob
            best_options = options[:k] # place top candidates in beam
        best_options.sort(key = lambda x: x[0], reverse=True)
        return best_options

    def get_attn_dist(self, src, trg):
        '''Runs forward pass, also returns attention distribution'''
        if use_gpu: src = src.cuda()
        # Encode, Decode, Attend
        out_e, final_e = self.encoder(src)
        out_d, final_d = self.decoder(trg, final_e)
        context = self.attention(src, out_e, out_d)
        out_cat = torch.cat((out_d, context), dim=2)
        # Predict
        x = self.linear1(out_cat)
        x = self.dropout(self.tanh(x))
        x = self.linear2(x)
        # Visualize attention distribution
        attn_dist = self.attention.get_visualization(src, out_e, out_d)
        return x, attn_dist
