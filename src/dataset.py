import os

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pycocotools.coco import COCO

from vocab import BasicTokenizer

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
        self.ids = list(self.coco.anns.keys())
        self.tokenizer = tokenizer
        self.transform = transform

    """
    Returns:
        {
            img_id:         image id;                   int
            image:          tensor representing image;  torch.Tensor
            raw_caption:    raw captions;               str
            caption:        index of encoded tokens;    torch.Tensor
            length:         lengths of captions;        torch.Tensor
        }
    """
    def __getitem__(self, idx):
        coco = self.coco
        ann_id = self.ids[idx]
        raw_caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.img_path, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        caption = self.tokenizer.encode([raw_caption])
        length = caption.ne(self.tokenizer.padidx).sum(dim=1)

        return {"img_id": img_id, "image": image, "raw_caption": raw_caption, "caption": caption, "length": length}

    def __len__(self):
        return len(self.ids)


# for debugging
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dest = "/groups1/gaa50131/datasets/MSCOCO/annotations/captions_train2017.txt"
    tokenizer = BasicTokenizer(min_freq=5, max_len=30)
    tokenizer.from_textfile(dest)
    ds = CocoDataset("/groups1/gaa50131/datasets/MSCOCO", mode="train", tokenizer=tokenizer, transform=transform)
    for i in range(30):
        print(ds[i]["raw_caption"])
        print(ds[i]["caption"])
        print(ds[i]["image"].size())





