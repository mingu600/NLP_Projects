import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch.nn import Parameter
from torch.nn.utils import clip_grad_norm
import torchtext
from tqdm import tqdm
import random
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devid", type=int, default=-1)

    parser.add_argument("--model", choices=["NnLm", "LstmLm"], default="LstmLm")
    parser.add_argument("--nhid", type=int, default=256)
    parser.add_argument("--nlayers", type=int, default=1)

    parser.add_argument("--tieweights", action="store_true")
    parser.add_argument("--maxnorm", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=0)

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--optim", choices=["SGD", "Adam"], default="Adam")

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lrd", type=float, default=0.25)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--bptt", type=int, default=32)
    parser.add_argument("--clip", type=float, default=5)

    # Adam parameters
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)

    # SGD parameters
    parser.add_argument("--mom", type=float, default=0.99)
    parser.add_argument("--dm", type=float, default=0)
    parser.add_argument("--nonag", action="store_true", default=False)

    return parser.parse_args()

args = parse_args()

if args.devid >= 0:
    torch.backends.cudnn.enabled = False
    print("Cudnn is enabled: {}".format(torch.backends.cudnn.enabled))


TEXT = torchtext.data.Field()
train, valid, test = torchtext.datasets.LanguageModelingDataset.splits(
    path=".",
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)
    #path="data/",
    #train="train.txt", validation="valid.txt", test="test.txt", text_field=TEXT)

TEXT.build_vocab(train)
padidx = TEXT.vocab.stoi["<pad>"]

train_iter, valid_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, valid, test), batch_size=args.bsz, device=args.devid, bptt_len=args.bptt, repeat=False)

class Lm(nn.Module):
    'Basic Language model'
    def __init__(self):
        super(Lm, self).__init__()

    def train_epoch(self, iter, loss, optimizer):
        self.train()
        train_loss = 0
        nwords = 0
        hid = None
        for batch in tqdm(iter):
            optimizer.zero_grad()
            x = batch.text
            y = batch.target
            out, hid = model(x, hid if hid is not None else None)
            bloss = loss(out.view(-1, model.vsize), y.view(-1))
            bloss.backward()
            train_loss += bloss
            nwords += y.ne(padidx).int().sum()
            if args.clip > 0:
                clip_grad_norm(self.parameters(), args.clip)
            optimizer.step()
        return train_loss.data[0], nwords.data[0]

    def validate(self, iter, loss):
        self.eval()
        valid_loss = 0
        nwords = 0
        hid = None
        for batch in iter:
            x = batch.text
            y = batch.target
            out, hid = model(x, hid if hid is not None else None)
            valid_loss += loss(out.view(-1, model.vsize), y.view(-1))
            nwords += y.ne(padidx).int().sum()
        return valid_loss.data[0], nwords.data[0]

class NnLm(Lm):
    """ Feedforward neural network LM, pretends each bptt sequence is a sentence. """
    def __init__(self, vocab, nhid, kW=3, nlayers=1, dropout=0, tieweights=True):
        super(NnLm, self).__init__()
        self.vsize = len(vocab.itos)
        self.kW = kW
        self.nhid = nhid
        self.lut = nn.Embedding(self.vsize, nhid, max_norm=args.maxnorm)
        self.conv = nn.Conv1d(nhid, nhid, kW, stride=1)
        self.drop = nn.Dropout(dropout)
        m = []
        for i in range(nlayers-1):
            m.append(nn.Linear(nhid, nhid))
            m.append(nn.Tanh())
            m.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*m)
        self.proj = nn.Linear(nhid, self.vsize)
        if tieweights:
            self.proj.weight = self.lut.weight

    def forward(self, input, hid):
        emb = self.lut(input)
        # T = time, N = batch, H = hidden
        T, N, H = emb.size()
        pad = V(emb.data.new(self.kW-1, N, H))
        pad.data.fill_(0)
        pad.requires_grad = False
        emb = torch.cat([pad, emb], 0)
        # Conv wants N(1) x H(2) x T(0), but our input is T(0) x N(1) x H(2)
        # Then we have to convert back to T(2) x N(0) x H(1) from N(0) x H(1) x T(2)
        hid = self.conv(emb.permute(1,2,0)).permute(2,0,1)
        return self.proj(self.mlp(hid)), hid


class LstmLm(Lm):
    def __init__(self, vocab, nhid, nlayers=1, dropout=0, tieweights=True):
        super(LstmLm, self).__init__()
        self.vsize = len(vocab.itos)
        self.lut = nn.Embedding(self.vsize, nhid, max_norm=args.maxnorm)
        self.rnn = nn.LSTM(
            input_size=nhid,
            hidden_size=nhid,
            num_layers=nlayers,
            dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(nhid, self.vsize)
        if tieweights:
            # See https://arxiv.org/abs/1608.05859
            # Seems to improve ppl by 13%.
            self.proj.weight = self.lut.weight

    def forward(self, input, hid):
        emb = self.lut(input)
        hids, hid = self.rnn(emb, hid)
        return self.proj(self.drop(hids)), tuple(map(lambda x: x.detach(), hid))


if __name__ == "__main__":
    models = {model.__name__: model for model in [NnLm, LstmLm]}
    model = models[args.model](
        TEXT.vocab, args.nhid, nlayers=args.nlayers, dropout=args.dropout, tieweights=args.tieweights)
    print(model)
    if args.devid >= 0:
        model.cuda(args.devid)

    # We do not want to give the model credit for predicting padding symbols,
    # this can decrease ppl a few points.
    weight = torch.FloatTensor(model.vsize).fill_(1)
    weight[padidx] = 0
    if args.devid >= 0:
        weight = weight.cuda(args.devid)
    loss = nn.CrossEntropyLoss(weight=V(weight), size_average=False)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "Adam":
        optimizer = optim.Adam(
            params, lr = args.lr, weight_decay = args.wd, betas=(args.b1, args.b2))
    elif args.optim == "SGD":
        optimizer = optim.SGD(
            params, lr = args.lr, weight_decay = args.wd,
            nesterov = not args.nonag, momentum = args.mom, dampening = args.dm)
    schedule = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=1, factor=args.lrd, threshold=1e-3)

    for epoch in range(args.epochs):
        print("Epoch {}, lr {}".format(epoch, optimizer.param_groups[0]['lr']))
        train_loss, train_words = model.train_epoch(
            iter=train_iter, loss=loss, optimizer=optimizer)
        valid_loss, valid_words = model.validate(valid_iter, loss)
        schedule.step(valid_loss)
        print("Train: {}, Valid: {}".format(
            math.exp(train_loss / train_words), math.exp(valid_loss / valid_words)))

    test_loss, test_words = model.validate(test_iter, loss)
    print("Test: {}".format(math.exp(test_loss / test_words)))
    model.generate_predictions()
    torch.save(model.cpu(), model.__class__.__name__ + ".pth")
