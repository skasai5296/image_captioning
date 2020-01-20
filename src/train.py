import sys, os
import argparse
import torch
import shutil
import argparse
from pprint import pprint
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import torchvision.transforms as transforms
import yaml
from addict import Dict
from nlgeval import NLGEval

from dataset import CocoDataset, train_collater, val_collater
from vocab import BasicTokenizer
from utils import Logger, AverageMeter, Timer, BleuComputer, ModelSaver
from simp_decoder import Captioning_Simple
from attn_decoder import Captioning_Attention

"""
Train for a single epoch.
Args:
    train_iterator:     DataLoader for training
    model:              Captioning model
    optimizer:          Optimizer
    criterion:          Loss function
    device:             CPU or GPU
    tb_logger:          TensorBoard logger
    ep:                 epoch
"""
def train_epoch(train_iterator, model, optimizer, criterion, device, tb_logger, ep):
    epoch_timer = Timer()
    model.train()
    m = model.module if hasattr(model, "module") else model
    # whether or not use doubly stochastic attention
    dsflag = isinstance(m, Captioning_Attention)
    losses = {}
    for it, data in enumerate(train_iterator):
        image = data["image"]
        caption = data["caption"]
        length = data["length"]
        image = image.to(device)
        caption = caption.to(device)
        length = length.to(device)

        optimizer.zero_grad()
        if dsflag:
            decoded, alphas = model(image, caption, length)
            losses["DSLoss"] = ((1. - alphas.sum(dim=2)) ** 2).mean()
        else:
            decoded = model(image, caption, length)
        losses["NLLLoss"] = criterion(decoded, caption[:, 1:])
        lossstr = ""
        cumloss = 0
        for loss_name, loss in losses.items():
            cumloss += loss
            tb_logger.add_scalar("loss/{}".format(loss_name), loss.item(), ep*len(train_iterator)+it)
            lossstr += " {}: {:.6f} |".format(loss_name, loss.item())
        cumloss.backward()
        optimizer.step()

        if it % 10 == 9:
            logging.info("epoch {} | iter {} / {} |{}".format(epoch_timer, it+1, len(train_iterator), lossstr))

"""
Validates and computes NLP metrics
Args:
    val_iterator:       DataLoader for validation
    model:              Captioning model
    tokenizer:          Tokenizer
    evaluator:          NLGEval instance for computing metrics
    device:             CPU or GPU
"""
def validate(val_iterator, model, tokenizer, evaluator, device):
    gt_list = []
    ans_list = []
    val_timer = Timer()
    model.eval()
    m = model.module if hasattr(model, "module") else model
    dsflag = isinstance(m, Captioning_Attention)
    for it, data in enumerate(val_iterator):
        image = data["image"]
        raw_caption = data["raw_caption"]
        image = image.to(device)

        with torch.no_grad():
            if dsflag:
                decoded, _ = m.sample(image)
            else:
                decoded = m.sample(image)
            # TODO: implement beamsearch
            decoded = torch.argmax(decoded, dim=1)
        generated = tokenizer.decode(decoded)

        gt_list.extend(raw_caption)
        ans_list.extend(generated)
        if it % 10 == 9:
            logging.info("validation {} | iter {} / {}".format(val_timer, it+1, len(val_iterator)))
            gts = [c[0] for c in raw_caption[:5]]
            hyps = generated[:5]
            for gt, hyp in zip(gts, hyps):
                logging.debug("ground truth: {}, sampled: {}".format(gt, hyp))
    metrics = {}
    logging.info("---METRICS---")
    metrics = evaluator.compute_metrics(ref_list=[[sen.strip() for sen in refs] for refs in zip(*gt_list)], hyp_list=ans_list)
    for k, v in metrics.items():
        logging.info("{}:\t\t{}".format(k, v))
    logging.info("---METRICS---")
    return metrics


if __name__ == "__main__":

    global_timer = Timer()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to configuration yml file')
    parser.add_argument('--resume', action="store_true", help='denotes if to continue training, will use config')
    parser.add_argument('--loglevel', type=str, default="debug", help='denotes log level, should be one of [debug|info|warning|error|critical]')
    opt = parser.parse_args()
    CONFIG = Dict(yaml.safe_load(open(opt.config)))

    numeric_level = getattr(logging, opt.loglevel.upper())
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: {}".format(numeric_level))
    outdir = os.path.join(CONFIG.result_dir, CONFIG.config_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    logging.basicConfig(filename=os.path.join(outdir, "{}.log".format(opt.loglevel.lower())), level=numeric_level)

    tb_logdir = os.path.join(CONFIG.log_dir, CONFIG.config_name)
    if not os.path.exists(tb_logdir):
        os.makedirs(tb_logdir)
    tb_logger = SummaryWriter(log_dir=tb_logdir)

    logging.debug("Initializing tokenizer and loading vocabulary from {} ...".format(os.path.join(CONFIG.data_path, CONFIG.caption_file_path)))
    tokenizer = BasicTokenizer(min_freq=CONFIG.min_freq, max_len=CONFIG.max_len)
    tokenizer.from_textfile(os.path.join(CONFIG.data_path, CONFIG.caption_file_path))
    logging.debug("done!")

    logging.debug("Initializing Dataset...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dset = CocoDataset(CONFIG.data_path, mode="train", tokenizer=tokenizer, transform=transform)
    train_loader = DataLoader(train_dset, batch_size=CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_worker, pin_memory=True, collate_fn=train_collater)
    val_dset = CocoDataset(CONFIG.data_path, mode="val", tokenizer=tokenizer, transform=transform)
    val_loader = DataLoader(val_dset, batch_size=CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_worker, pin_memory=True, collate_fn=val_collater)
    logging.debug("done!")

    logging.debug("loading model...")
    if torch.cuda.is_available:
        device = torch.device("cuda")
        logging.debug("using {} GPU(s)".format(torch.cuda.device_count()))
    else:
        device = torch.device("cpu")
        logging.debug("using CPU")
    if CONFIG.attention:
        model = Captioning_Attention(cnn_type=CONFIG.cnn_arch, pretrained=True, spatial_size=CONFIG.spatial_size, emb_dim=CONFIG.emb_dim, memory_dim=CONFIG.memory_dim,
            vocab_size=len(tokenizer), max_seqlen=CONFIG.max_len, dropout_p=CONFIG.dropout_p, ss_prob=CONFIG.ss_prob, bos_idx=tokenizer.bosidx)
    else:
        model = Captioning_Simple(cnn_type=CONFIG.cnn_arch, pretrained=True, spatial_size=CONFIG.spatial_size, emb_dim=CONFIG.emb_dim, memory_dim=CONFIG.memory_dim,
            vocab_size=len(tokenizer), max_seqlen=CONFIG.max_len, dropout_p=CONFIG.dropout_p, ss_prob=CONFIG.ss_prob, bos_idx=tokenizer.bosidx)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG.lr, betas=(CONFIG.beta1, CONFIG.beta2))

    model_path = os.path.join(outdir, "best_score.ckpt")
    saver = ModelSaver(model_path, init_val=0)
    offset_ep = 1
    if opt.resume:
        offset_ep = saver.load_ckpt(model, optimizer, device)
        if offset_ep > CONFIG.max_epoch:
            logging.error("trying to restart at epoch {} while max training is set to {} epochs".format(offset_ep, CONFIG.max_epoch))
            sys.exit(1)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.padidx)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    logging.debug("done!")

    logging.debug("loading evaluator...")
    #evaluator = BleuComputer()
    evaluator = NLGEval(metrics_to_omit=["METEOR"]) # meteor has problems, so omit
    logging.debug("done!")

    for ep in range(offset_ep-1, CONFIG.max_epoch):
        logging.info("global {} | begin training for epoch {}".format(global_timer, ep+1))
        train_epoch(train_loader, model, optimizer, criterion, device, tb_logger, ep)
        logging.info("global {} | done with training for epoch {}, beginning validation".format(global_timer, ep+1))
        metrics = validate(val_loader, model, tokenizer, evaluator, device)
        for key, val in metrics.items():
            tb_logger.add_scalar("metrics/{}".format(key), val, ep+1)
        if "Bleu_4" in metrics.keys():
            saver.save_ckpt_if_best(model, optimizer, metrics["Bleu_4"])
        logging.info("global {} | end epoch {}".format(global_timer, ep+1))
    logging.info("done training!!")





