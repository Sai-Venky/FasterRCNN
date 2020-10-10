from __future__ import  absolute_import
from __future__ import  division
from data.voc_dataset import VOCDataset
from data.util import transform
from utils.config import opt


def inverse_normalize(img):
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255

class Dataset:
    def __init__(self, opt):
        self.voc = VOCDataset(opt.voc_data_dir)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.voc.get_example(idx)
        img, bbox, label, scale = transform((ori_img, bbox, label))
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.voc)