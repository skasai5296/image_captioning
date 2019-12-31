import sys, os
import time
import random
import csv
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.init as init
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu

class BleuComputer():
    def __init__(self, n=4):
        self.n = n

    def compute_metrics(self, ref_list, hyp_list):
        ref_list = [[sentence.split() for sentence in batch] for batch in ref_list]
        hyp_list = [sentence.split() for sentence in hyp_list]
        out = {}
        for i in range(self.n):
            weights = [0, 0, 0, 0]
            weights[i] = 1
            out["BLEU-{}".format(i+1)] = corpus_bleu(ref_list, hyp_list, weights)
        return out

class ModelSaver():
    def __init__(self, path, init_val=0):
        self.path = path
        self.best = init_val

    def load_ckpt(self, model, optimizer):
        if os.path.exists(self.path):
            logging.info(f"loading model from {self.path}")
            model.load_state_dict(torch.load(self.path), map_location="cpu")
        else:
            logging.error(f"{self.path} does not exist, not loading")

    def save_ckpt_if_best(self, model, optimizer, metric):
        if metric > self.best:
            logging.info(f"score {metric} is better than previous best score of {self.best}, saving to {self.path}")
            save = { "optimizer": optimizer.state_dict() }
            if hasattr(model, "module"):
                save["model"] = model.module.state_dict()
                torch.save(save, self.path)
            else:
                save["model"] = model.state_dict()
                torch.save(save, self.path)
            self.best = metric
        else:
            logging.info(f"score {metric} is not better than previous best score of {self.best}, not saving")


class Logger():

    def __init__(self, path, header):
        path = Path(path)
        self.log_file = path.open('w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}: {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)

class Timer():
    """Computes and stores the time"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.begin = time.time()

    def __str__(self):
        return sec2str(time.time()-self.begin)

def sec2str(sec):
    if sec < 60:
        return "time elapsed: {:02d}s".format(int(sec))
    elif sec < 3600:
        min = int(sec / 60)
        sec = int(sec - min * 60)
        return "time elapsed: {:02d}m{:02d}s".format(min, sec)
    elif sec < 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        return "time elapsed: {:02d}h{:02d}m{:02d}s".format(hr, min, sec)
    elif sec < 365 * 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        dy = int(hr / 24)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        hr = int(hr - dy * 24)
        return "time elapsed: {:02d} days, {:02d}h{:02d}m{:02d}s".format(dy, hr, min, sec)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
