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
    pretrained:     Use pretrained model
    spatial_size:   Size (one side) of output feature
"""
class ImageEncoder(nn.Module):
    def __init__(self, cnn_type, pretrained, spatial_size):
        super().__init__()
        assert cnn_type.startswith("resnet")
        cnn = getattr(torchvision.models, cnn_type)(pretrained)
        self.out_size = cnn.fc.in_features
        self.cnn = nn.Sequential(*list(cnn.children())[:-2])
        if not pretrained:
            self.cnn.apply(weight_init)
        self.pool = nn.AdaptiveAvgPool2d((spatial_size, spatial_size))

    """
    Args:
        torch.Tensor image:         (bs x 3 x H x W)
    Returns:
        torch.Tensor feature:       (bs x spatial_size*spatial_size x out_size)
    """
    def forward(self, image):
        bs = image.size(0)
        feature = self.cnn(image)
        feature = self.pool(feature)
        return feature.reshape(bs, self.out_size, -1).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, ft_dim, rnn_dim, attn_dim):
        super().__init__()
        self.enc_attn = nn.Linear(ft_dim, attn_dim)
        self.dec_attn = nn.Linear(rnn_dim, attn_dim)
        self.attn = nn.Linear(attn_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    """
    Args:
        torch.Tensor feature:               (bs x * x ft_dim)
        torch.Tensor memory:                (bs x rnn_dim)
    Returns:
        torch.Tensor attn_weights:          (bs x *)
        torch.Tensor weighted_feature:      (bs x ft_dim)
    """
    def forward(self, feature, memory):
        # encoded_feature: (bs x spatial_size*spatial_size x attn_dim)
        encoded_feature = self.enc_attn(feature)
        # encoded_memory: (bs x 1 x attn_dim)
        encoded_memory = self.dec_attn(memory).unsqueeze(1)
        # attn_weights: (bs x spatial_size*spatial_size)
        attn_weights = self.attn(self.relu(encoded_feature + encoded_memory)).squeeze(-1)
        attn_weights = self.softmax(attn_weights)
        weighted_feature = (feature * attn_weights.unsqueeze(-1)).sum(dim=1)
        return attn_weights, weighted_feature


"""
RNN decoder with attention for captioning, Show, Attend and Tell
Args:
    feature_dim:    dimension of image feature
    spatial_size:   spatial size of feature (one side)
    emb_dim:        dimension of word embeddings
    memory_dim:     dimension of LSTM memory and attention
    vocab_size:     vocabulary size
    max_seqlen:     max sequence size
    dropout_p:      dropout probability for LSTM memory
    ss_prob:        scheduled sampling rate, 0 for teacher forcing and 1 for free running
"""
class AttentionDecoder(nn.Module):
    def __init__(self, feature_dim, spatial_size, emb_dim, memory_dim, vocab_size, max_seqlen, dropout_p, ss_prob, bos_idx):
        super().__init__()
        self.max_seqlen = max_seqlen
        self.ss_prob = ss_prob
        self.bos_idx = bos_idx

        self.init_h = nn.Linear(feature_dim * spatial_size * spatial_size, memory_dim)
        self.init_c = nn.Linear(feature_dim * spatial_size * spatial_size, memory_dim)
        self.attention = Attention(feature_dim, memory_dim, memory_dim)
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTMCell(emb_dim + feature_dim, memory_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(memory_dim, vocab_size)

        self.init_h.apply(weight_init)
        self.init_c.apply(weight_init)
        self.rnn.apply(weight_init)
        self.emb.apply(weight_init)
        self.linear.apply(weight_init)

    """
    Args:
        torch.Tensor feature:       (bs x spatial_size*spatial_size x feature_dim), torch.float
        torch.Tensor caption:       (bs x max_seqlen), torch.long
        torch.Tensor length:        (bs), torch.long
    Returns:
        torch.Tensor out:           (bs x vocab_size x max_seqlen-1), contains logits
    """
    def forward(self, feature, caption, length):
        bs = feature.size(0)
        # hn, cn: (bs x memory_dim)
        hn = self.init_h(feature.reshape(bs, -1))
        cn = self.init_c(feature.reshape(bs, -1))
        # caption: (bs x max_seqlen x emb_dim)
        caption = self.emb(caption)
        xn = caption[:, 0, :]
        out = []
        for step in range(1, self.max_seqlen):
            # alpha: (bs x spatial_size*spatial_size)
            # weighted_feature: (bs x feature_dim)
            alpha, weighted_feature = self.attention(feature, hn)
            # hn, cn: (bs x memory_dim)
            hn, cn = self.rnn(torch.cat([xn, weighted_feature], dim=1), (hn, cn))
            # on: (bs x vocab_size)
            on = self.linear(self.dropout(hn))
            out.append(on)
            # xn: (bs x emb_dim)
            xn = self.emb(on.argmax(dim=1)) if np.random.uniform() < self.ss_prob else caption[:, step, :]
        # out: (bs x vocab_size x max_seqlen-1)
        out = torch.stack(out, dim=-1)
        return out

    def sample(self, feature):
        bs = feature.size(0)
        # hn, cn: (bs x memory_dim)
        hn = self.init_h(feature.reshape(bs, -1))
        cn = self.init_c(feature.reshape(bs, -1))
        # xn: (bs x emb_dim)
        xn = self.emb(torch.full((bs,), self.bos_idx, dtype=torch.long, device=feature.device))
        out = []
        for step in range(1, self.max_seqlen):
            # alpha: (bs x spatial_size*spatial_size)
            # weighted_feature: (bs x feature_dim)
            alpha, weighted_feature = self.attention(feature, hn)
            # hn, cn: (bs x memory_dim)
            hn, cn = self.rnn(torch.cat([xn, weighted_feature], dim=1), (hn, cn))
            # on: (bs x vocab_size)
            on = self.linear(self.dropout(hn))
            out.append(on)
            # xn: (bs x emb_dim)
            xn = self.emb(on.argmax(dim=1))
        # out: (bs x vocab_size x max_seqlen-1)
        out = torch.stack(out, dim=-1)
        return out

class Captioning_Attention(nn.Module):
    def __init__(self, cnn_type, pretrained, spatial_size, emb_dim, memory_dim, vocab_size, max_seqlen, dropout_p, ss_prob, bos_idx):
        super().__init__()
        self.cnn = ImageEncoder(cnn_type, pretrained, spatial_size)
        feature_dim = self.cnn.out_size
        self.rnn = AttentionDecoder(feature_dim, spatial_size, emb_dim, memory_dim, vocab_size, max_seqlen, dropout_p, ss_prob, bos_idx)

    def forward(self, image, caption, length):
        feature = self.cnn(image)
        return self.rnn(feature, caption, length)

    def sample(self, image):
        feature = self.cnn(image)
        return self.rnn.sample(feature)


