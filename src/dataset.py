import os
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from pycocotools.coco import COCO

from vocab import BasicTokenizer

import accimage
torchvision.set_image_backend("accimage")
from PIL import Image

class CocoDataset(Dataset):
    """
    Args:
        root_path:      path to COCO dataset
        mode:           train or val
        tokenizer:      train or val
        transform:      augmentation on images, likely a transforms.Compose object
    """
    def __init__(self, root_path: str, mode: str, tokenizer, transform = None):
        assert mode in ("train", "val")
        assert isinstance(tokenizer, BasicTokenizer)
        self.img_path = os.path.join(root_path, "{}2017".format(mode))
        self.coco = COCO(os.path.join(root_path, "annotations", "captions_{}2017.json").format(mode))
        self.img_ids = self.coco.getImgIds()
        self.ann_ids = [self.coco.getAnnIds(id) for id in self.img_ids]
        self.tokenizer = tokenizer
        self.transform = transform

    """
    Returns:
        {
            img_id:         image id;                   int
            image:          tensor representing image;  likely torch.Tensor (3 x 224 x 224) after transform
            raw_caption:    raw captions;               list of str
            caption:        index of encoded tokens;    torch.Tensor (5 x max_seqlen)
            length:         lengths of captions;        torch.Tensor (5)
        }
    """
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_id = self.ann_ids[idx][:5]
        path = self.coco.loadImgs(img_id)[0]['file_name']

        raw_caption = [obj['caption'] for obj in self.coco.loadAnns(ann_id)]
        image = accimage.Image(os.path.join(self.img_path, path))
        if self.transform is not None:
            image = self.transform(image)
        caption = self.tokenizer.encode(raw_caption)
        length = caption.ne(self.tokenizer.padidx).sum(dim=1)

        return {"img_id": img_id, "image": image, "raw_caption": raw_caption, "caption": caption, "length": length}

    def __len__(self):
        return len(self.img_ids)

"""
Returns:
    {
        img_id:         image id;                   list of int
        image:          tensor representing image;  torch.Tensor (N x 3 x 224 x 224)
        raw_caption:    raw captions;               list of str
        caption:        index of encoded tokens;    torch.Tensor (N x max_seqlen)
        length:         lengths of captions;        torch.Tensor (N)
    }
"""
def train_collater(data):
    img_id = []
    image = []
    raw_caption = []
    caption = []
    length = []
    for x in data:
        idx = np.random.randint(len(x["raw_caption"]))
        img_id.append(x["img_id"])
        image.append(x["image"])
        raw_caption.append(x["raw_caption"][idx])
        caption.append(x["caption"][idx])
        length.append(x["length"][idx])
    image = torch.stack(image)
    caption = torch.stack(caption)
    length = torch.stack(length)
    return {"img_id": img_id, "image": image, "raw_caption": raw_caption, "caption": caption, "length": length}

"""
Returns:
    {
        img_id:         image id;                   list of int
        image:          tensor representing image;  torch.Tensor (N x 3 x 224 x 224)
        raw_caption:    raw captions;               list of list of str
        caption:        index of encoded tokens;    torch.Tensor (N x 5 x max_seqlen)
        length:         lengths of captions;        torch.Tensor (N x 5)
    }
"""
def val_collater(data):
    img_id = []
    image = []
    raw_caption = []
    caption = []
    length = []
    for x in data:
        img_id.append(x["img_id"])
        image.append(x["image"])
        raw_caption.append(x["raw_caption"])
        caption.append(x["caption"])
        length.append(x["length"])
    image = torch.stack(image)
    caption = torch.stack(caption)
    length = torch.stack(length)
    return {"img_id": img_id, "image": image, "raw_caption": raw_caption, "caption": caption, "length": length}



# for debugging
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dest = "/home/seito/hdd/dsets/coco/annotations/captions_train2017.txt"
    tokenizer = BasicTokenizer(min_freq=5, max_len=30)
    tokenizer.from_textfile(dest)
    ds = CocoDataset("/home/seito/hdd/dsets/coco", mode="train", tokenizer=tokenizer, transform=transform)
    for i in range(10):
        print(ds[i]["image"].size())
        print(ds[i]["raw_caption"])
        print(ds[i]["caption"])
        print(ds[i]["length"])





