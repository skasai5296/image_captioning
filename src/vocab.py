import sys, os
import time
import json
import pickle

import torch
import torchtext
import spacy

spacy.load('en_core_web_sm')

class BasicTokenizer():
    """
    Args:
        min_freq:   int;     word frequency threshold when building vocabulary
        max_len:    int;     max length when building tensors
    """
    def __init__(self, min_freq, max_len):
        self.min_freq = min_freq
        self.max_len = max_len
        self.field = torchtext.data.Field(sequential=True, init_token="<bos>", eos_token="<eos>", lower=True, fix_length=self.max_len, tokenize="spacy", batch_first=True)

    """
    Build vocabulary from textfile.
    Sentences are separated by '\n'
    Args:
        textfile:   str;    path to textfile containing vocabulary
    """
    def from_textfile(self, textfile):
        print("loading vocabulary from {} ...".format(textfile))
        with open(textfile, 'r') as f:
            sentences = f.readlines()
        sent_proc = list(map(self.field.preprocess, sentences))
        self.field.build_vocab(sent_proc, min_freq=self.min_freq)
        self.len = len(self.field.vocab)
        self.padidx = self.field.vocab.stoi["<pad>"]
        print("done!")

    """
    Tokenize and numericalize a single or batched sentence.
    Converts into torch.Tensor from list of captions.
    Args:
        sentence_batch:     list of str or str; captions put together in a list
    Returns:
        out:                torch.Tensor;       (max_len) if input is string, (batch_size x max_len) if list
    """
    def encode(self, sentence):
        assert isinstance(sentence, list) or isinstance(sentence, str)
        strflag = isinstance(sentence, str)
        if strflag:
            sentence = [sentence]
        preprocessed = list(map(self.field.preprocess, sentence))
        out = self.field.process(preprocessed)
        if strflag:
            out = out.squeeze(0)
        return out

    """
    Reverse conversion from torch.Tensor to a list of captions.
    Args:
        ten:    torch.Tensor (bs x seq)
    Returns:
        out:    list of str
    """
    def decode(self, ten):
        assert isinstance(ten, torch.Tensor)
        assert ten.dtype == torch.long
        assert ten.dim() in (1, 2)
        if ten.dim() == 1:
            ten = ten.unsqueeze(0)
        length = ten.ne(self.padidx).sum(dim=1)
        ten = ten.tolist()
        out = []
        for n, idxs in zip(length, ten):
            tokenlist = [self.field.vocab.itos[idx] for idx in idxs[1:n-1]]
            out.append(" ".join(tokenlist))
        return out

    def __len__(self):
        return self.len

"""
Takes annotation file and creates a .txt file
containing the captions
Args:
    jsonfile:   json file with coco caption annotations
    dst:        path to text file to write captions
"""
def cococap2txt(jsonfile: str, dst: str):
    sentences = []
    with open(jsonfile, 'r') as f:
        alldata = json.load(f)
    for ann in alldata["annotations"]:
        sentences.append(ann["caption"].strip())
    with open(dst, 'w+') as f:
        f.write("\n".join(sentences))

# for debugging and creation of text file
if __name__ == '__main__':
    root = "/home/seito/hdd/dsets/coco"
    file = os.path.join(root, "annotations/captions_train2017.json")
    dest = os.path.join(root, "annotations/captions_train2017.txt")
    # first time only
    if not os.path.exists(dest):
        cococap2txt(file, dest)
    tokenizer = BasicTokenizer(min_freq=5, max_len=30)
    tokenizer.from_textfile(dest)

    sentence1 = ["hello world this is my friend Alex.", "the cat and the rat sat on a mat."]
    ten = tokenizer.encode(sentence1)
    print(ten)
    sent = tokenizer.decode(ten)
    print(sent)

    sentence2 = "hello world this is my friend Alex."
    ten = tokenizer.encode(sentence2)
    print(ten)
    sent = tokenizer.decode(ten)
    print(sent)


