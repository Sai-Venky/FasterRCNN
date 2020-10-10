from utils.anchors import Anchors
from utils.helper import init_params, totensor
import numpy as np
import torch as t
from utils.config import opt
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg16
from torchvision.ops import RoIPool


def get_rcnn_vgg16():
    model = vgg16(not opt.load_path)

    features = list(model.features)[:30]
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    
    classifier = list(model.classifier)
    del classifier[6]
    classifier = nn.Sequential(*classifier)

    return nn.Sequential(*features), classifier


class VGG16RoIHead(nn.Module):

    def __init__(self, n_class, spatial_scale, classifier, roi_size = 7):
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.roi = RoIPool( (roi_size, roi_size), spatial_scale)
        self.pred_locs = nn.Linear(4096, n_class * 4)
        self.pred_scores = nn.Linear(4096, n_class)

        init_params(self.pred_locs, 0, 0.001)
        init_params(self.pred_scores, 0, 0.01)

    def forward(self, x, rois, roi_indices):
        roi_indices = totensor(roi_indices).float()
        rois = totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_pred_locs = self.pred_locs(fc7)
        roi_pred_scores = self.pred_scores(fc7)
        return roi_pred_locs, roi_pred_scores

