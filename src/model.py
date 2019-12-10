import sys, os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision

from utils import weight_init

"""
CNN encoder implemented with built-in PyTorch ResNet
Args:
    cnn_type:       Image encoder
    pool:           Whether or not to spatially pool features. False for Show, Attend and Tell
    pretrained:     Use pretrained model
"""
class ImageEncoder(nn.Module):
    def __init__(self, cnn_type="resnet18", pool=False, pretrained=True):
        super(ImageEncoder, self).__init__()
        assert cnn_type.startswith("resnet")
        cnn = getattr(torchvision.models, cnn_type)(pretrained)
        self.pool = pool
        if self.pool:
            self.cnn = nn.Sequential(*list(cnn.children())[:-1])
        else:
            self.cnn = nn.Sequential(*list(cnn.children())[:-2])
        if not pretrained:
            self.cnn.apply(weight_init)

    """
    Args:
        torch.Tensor x:         (bs x 3 x H x W)
    Returns:
        torch.Tensor out:       (bs x 512) if pool else (bs x 512 x H/32 x W/32)
    """
    def forward(self, x):
        out = self.cnn(x)
        if self.pool:
            out = out.squeeze(-1).squeeze(-1)
        return out

"""
RNN decoder for captioning, Google NIC
Args:
    feature_dim:    dimension of image features
    emb_dim:        dimension of word embeddings
    memory_dim:     dimension of LSTM memory
    vocab_size:     vocabulary size
    max_seqlen:     max sequence size
    dropout_p:      dropout probability for LSTM memory
    ss_prob:        scheduled sampling rate, 0 for teacher forcing and 1 for free running
"""
class SimpleDecoder(nn.Module):
    def __init__(self, feature_dim, emb_dim, memory_dim, vocab_size, max_seqlen, dropout_p, ss_prob):
        super(SimpleDecoder, self).__init__()
        self.emb_dim = emb_dim
        self.memory_dim = memory_dim
        self.vocab_size = vocab_size
        self.max_seqlen = max_seqlen
        self.ss_prob = ss_prob

        self.linear1 = nn.Linear(feature_dim, self.emb_dim)
        self.emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.rnn = nn.LSTMCell(self.emb_dim, self.memory_dim)
        self.linear2 = nn.Linear(self.memory_dim, self.vocab_size)
        self.dropout = nn.Dropout(dropout_p)

        self.linear1.apply(weight_init)
        self.rnn.apply(weight_init)
        self.emb.apply(weight_init)
        self.linear2.apply(weight_init)

    """
    Args:
        torch.Tensor image:         (bs x 512), torch.float
        torch.Tensor caption:       (bs x max_seqlen), torch.long
        torch.Tensor length:        (bs), torch.long
    Returns:
        torch.Tensor out:           (bs x vocab_size x max_seqlen), contains logits
    """
    def forward(self, image, caption, length):
        bs = image.size(0)
        # (bs x 512) -> (bs x emb_dim)
        feature = self.linear1(image)
        # (bs x max_seqlen) -> (bs x max_seqlen x emb_dim)
        caption = self.emb(caption)
        # hn: (bs x memory_dim)
        hn, cn = self.rnn(feature)
        # on: (bs x vocab_size)
        on = self.linear2(self.dropout(hn))
        out = [on]
        # xn: (bs x emb_dim)
        xn = self.emb(on.argmax(dim=1))
        for step in range(self.max_seqlen-1):
            # xn: (bs x emb_dim)
            if np.random.uniform() < self.ss_prob:
                xn = caption[:, step, :]
            # hn: (bs x memory_dim)
            hn, cn = self.rnn(xn, (hn, cn))
            # on: (bs x vocab_size)
            on = self.linear2(self.dropout(hn))
            # xn: (bs x emb_dim)
            xn = self.emb(on.argmax(dim=1))
            out.append(on)
        # out: (bs x vocab_size x max_seqlen)
        out = torch.stack(out, dim=-1)
        return out

    def sample(self, image):
        bs = image.size(0)
        # (bs x 512) -> (bs x emb_dim)
        feature = self.linear1(image)
        # hn: (bs x memory_dim)
        hn, cn = self.rnn(feature)
        # on: (bs x vocab_size)
        on = self.linear2(self.dropout(hn))
        out = [on]
        # xn: (bs x emb_dim)
        xn = self.emb(on.argmax(dim=1))
        for step in range(self.max_seqlen-1):
            # hn: (bs x memory_dim)
            hn, cn = self.rnn(xn, (hn, cn))
            # on: (bs x vocab_size)
            on = self.linear2(hn)
            # xn: (bs x emb_dim)
            xn = self.emb(on.argmax(dim=1))
            out.append(on)
        # out: (bs x vocab_size x max_seqlen)
        out = torch.stack(out, dim=-1).max(dim=1)[0]
        return out


class Attention(nn.Module):
    def __init__(self,
                 encoder_dim,
                 decoder_dim,
                 attention_dim
                 ):
        super(Attention, self).__init__()
        self.input_enc = nn.Linear(encoder_dim, attention_dim)
        self.output_enc = nn.Linear(decoder_dim, attention_dim)
        self.att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    # feature : (batch_size, H*W, encoder_dim)
    # dec_out : (batch_size, decoder_dim)
    def forward(self, feature, dec_out):
        out1 = self.input_enc(feature)
        out2 = self.output_enc(dec_out)
        # out : (batch_size, H*W, attention_dim)
        out = self.relu(out1, out2.unsqueeze(1))
        # att : (batch_size, H*W)
        att = self.att(out).unsqueeze(-1)
        # alpha : (batch_size, H*W)
        alpha = self.softmax(att)
        # weighted : (batch_size, encoder_dim)
        weighted = (feature * alpha.unsqueeze(-1)).sum(dim=1)
        return alpha, weighted


class LSTMCaptioning_Attention(nn.Module):
    def __init__(self,
                 enc_dim=512,
                 lstm_memory=512,
                 attention_size=512,
                 emb_size=512,
                 vocab_size=100,
                 max_seqlen=20,
                 num_layers=1,
                 dropout_p=0.1
                 ):
        super(LSTMCaptioning_Attention, self).__init__()
        self.enc_dim = enc_dim
        self.lstm_memory = lstm_memory
        self.attention_size = attention_size
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.max_seqlen = max_seqlen
        self.num_layers = num_layers

        self.attention = Attention(enc_dim, lstm_memory, attention_size)
        self.init_h = nn.Linear(enc_dim, lstm_memory)
        self.init_c = nn.Linear(enc_dim, lstm_memory)
        self.gate = nn.Linear(enc_dim, lstm_memory)
        self.emb = nn.Embedding(vocab_size, emb_size)
        #self.rnn = nn.LSTM(self.emb_size, self.lstm_memory, self.num_layers, batch_first=True, dropout=dropout_p)
        self.rnn = nn.LSTMCell(emb_size+enc_dim, lstm_memory)
        self.sigmoid = nn.Sigmoid()
        self.outlinear = nn.linear(lstm_memory, vocab_size)

    # THWC must be flattened for image feature
    # image_feature : (batch_size, T, H, W, enc_dim)
    # captions : (batch_size, seq_len)
    # returns : (batch_size, vocab_size, seq_len)
    def forward(self, image_feature, captions):
        bs = image_feature.size(0)
        dim = image_feature.size(-1)
        assert dim == self.enc_dim
        # feature : (batch_size, T*H*W, enc_dim)
        feature = image_feature.view(bs, -1, dim)
        pixnum = feature.size(1)

        # mean_ft : (batch_size, enc_dim)
        mean_ft = feature.mean(dim=1)
        # hn, cn : (batch_size, lstm_memory)
        hn = self.init_h(mean_ft)
        cn = self.init_c(mean_ft)

        outputlist = []
        # inputs : (batch_size, 1, emb_size)
        for idx in range(self.max_seqlen):
            weighted, alpha = self.attention(feature, hn)
            hiddens, cn = self.rnn(inputs, cn)
            # outputs : (batch_size, vocab_size)
            outputs = self.linear2(hiddens.squeeze(1))
            outputlist.append(outputs)
            # pred : (batch_size), LongTensor
            _, pred = outputs.max(1)
            # inputs : (batch_size, emb_size)
            inputs = self.emb(pred)
            inputs = inputs.unsqueeze(1)
        # sampled : (batch_size, vocab_size, max_seqlen)
        sampled = torch.stack(outputlist, 2)


        # captions : (batch_size, seq_len, emb_size)
        captions = self.emb(captions)
        # inseq : (batch_size, 1+seq_len, emb_size)
        inseq = torch.cat((feature, captions), dim=1)
        # hiddens : (batch_size, seq_len, lstm_memory)
        hiddens, _ = self.rnn(inseq[:, :-1, :])
        # outputs : (batch_size, vocab_size, seq_len)
        outputs = self.linear2(hiddens).transpose(1, 2)
        return outputs

    # image_feature : (batch_size, ft_size)
    # method : one of ['greedy', 'beamsearch']
    def sample(self, image_feature, method='greedy', init_state=None):
        outputlist = []
        # inputs : (batch_size, 1, emb_size)
        inputs = self.linear1(image_feature).unsqueeze(1)
        states = init_state
        for idx in range(self.max_seqlen):
            hiddens, states = self.rnn(inputs, states)
            # outputs : (batch_size, vocab_size)
            outputs = self.linear2(hiddens.squeeze(1))
            outputlist.append(outputs)
            # pred : (batch_size), LongTensor
            _, pred = outputs.max(1)
            # inputs : (batch_size, emb_size)
            inputs = self.emb(pred)
            inputs = inputs.unsqueeze(1)
        # sampled : (batch_size, vocab_size, max_seqlen)
        sampled = torch.stack(outputlist, 2)
        return sampled




