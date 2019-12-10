import sys, os
import time
import argparse
import torch
import shutil
import argparse
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import torchvision.transforms as transforms
from nlgeval import NLGEval
import yaml
from addict import Dict

from dataset import CocoDataset
from vocab import BasicTokenizer
from utils import Logger, AverageMeter
from model import ImageEncoder, SimpleDecoder

"""
Train for a single epoch.
Args:
    train_iterator:     DataLoader for training
    encoder:            Image encoder
    decoder:            Caption decoder
    optimizer:          Optimizer
    criterion:          Loss function
    device:             CPU or GPU
"""
def train_epoch(train_iterator, encoder, decoder, optimizer, criterion, device):
    loss_meter = AverageMeter("NLL Loss", ":.5f")
    for it, data in enumerate(train_iterator):
        image = data["image"]
        caption = data["caption"]
        length = data["length"]
        idx = torch.randint(5, size=(1, caption.size(1), 1))
        caption = torch.gather(caption, 1, idx.expand_as(caption))[:, 0, :]
        length = torch.gather(caption, 1, idx.squeeze(-1).expand_as(length))[:, 0]
        image.to(device)
        caption.to(device)
        length.to(device)

        optimizer.zero_grad()
        encoded = encoder(image)
        decoded = decoder(encoded, caption, length)
        loss = criterion(decoded, caption)
        loss.backward()
        optimizer.step()
        if it % 1 == 0:
            print("iter {:06d} / {:06d} \t nll loss: {:.5f}".format(it+1, len(train_iterator), loss.item()), flush=True)

"""
Validates and computes NLP metrics
Args:
    val_iterator:       DataLoader for validation
    encoder:            Image encoder
    decoder:            Caption decoder
    tokenizer:          Tokenizer
    evaluator:          NLGEval instance for computing metrics
    device:             CPU or GPU
"""
def validate(val_iterator, encoder, decoder, tokenizer, evaluator, device):
    gt_list = []
    ans_list = []
    for it, data in enumerate(val_iterator):
        image = data["image"]
        raw_caption = data["caption"]
        image.to(device)

        encoded = encoder(image)
        decoded = decoder.sample(encoded)
        generated = tokenizer.decode(decoded)

        gt_list.extend(raw_caption)
        ans_list.extend(generated)
        if it % 1 == 0:
            print("iter {} / {}".format(it+1, len(val_iterator)), flush=True)
    print("---METRICS---", flush=True)
    try:
        metrics = evaluator.compute_metrics(ref_list=[gt_list], hyp_list=ans_list)
        for k, v in metrics.items():
            print("{}:\t\t{}".format(k, v))
    except:
        metrics = None
        print("could not evaluate, some sort of error in NLGEval", flush=True)
    print("---METRICS---", flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../results/default/config.yml')
    parser.add_argument('--resume', action="store_true", help='denotes if to continue training, will use config')
    opt = parser.parse_args()

    CONFIG = Dict(yaml.safe_load(open(opt.config)))
    print("CONFIGURATIONS:")
    pprint(CONFIG)

    tokenizer = BasicTokenizer(min_freq=CONFIG.min_freq, max_len=CONFIG.max_len)
    tokenizer.from_textfile(os.path.join(CONFIG.data_path, CONFIG.caption_file_path))
    log = ["epoch", "Bleu-1", "Bleu-2", "Bleu-3", "Bleu-4", "METEOR", "CIDEr", "ROUGE_L"]
    metric_logger = Logger(os.path.join("../results", CONFIG.config_name, "stats.tsv"), log)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dset = CocoDataset(CONFIG.data_path, mode="train", tokenizer=tokenizer, transform=transform)
    train_loader = DataLoader(train_dset, batch_size=CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_worker)
    val_dset = CocoDataset(CONFIG.data_path, mode="val", tokenizer=tokenizer, transform=transform)
    val_loader = DataLoader(val_dset, batch_size=CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_worker)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading models...")
    encoder = ImageEncoder(cnn_type=CONFIG.cnn_arch, pool=True, pretrained=True)
    decoder = SimpleDecoder(feature_dim=CONFIG.feature_dim, emb_dim=CONFIG.emb_dim, memory_dim=CONFIG.memory_dim,
            vocab_size=len(tokenizer), max_seqlen=CONFIG.max_len, dropout_p=CONFIG.dropout_p, ss_prob=CONFIG.ss_prob)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=CONFIG.lr, betas=(CONFIG.beta1, CONFIG.beta2))
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.padidx)
    print("done!")
    print("loading evaluator...")
    # uses ~6G RAM!!
    evaluator = NLGEval()
    print("done!")

    for ep in range(CONFIG.max_epoch):
        print("begin training for epoch {}".format(ep+1))
        train_epoch(train_loader, encoder, decoder, optimizer, criterion, device)
        print("done with training for epoch {}, beginning validation".format(ep+1))
        metrics = validate(val_loader, encoder, decoder, tokenizer, evaluator, device)
        print("end validation for epoch {}".format(ep+1))
        metrics["epoch"] = ep+1
        metric_logger.log(metrics)
    print("done training!!")





