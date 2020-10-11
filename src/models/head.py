import numpy as np
import torch as t
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg16
from torchvision.ops import RoIPool

from utils.anchors import Anchors
from utils.helper import init_params, totensor
from utils.config import opt


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

    """
    FasterRCNN VGG-16 Head.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Arguments:
        n_class         :- The number of classes including the background.
        spatial_scale   :- Scale of the roi is resized.
        classifier      :- Two layer Linear ported from vgg16
        roi_size        :- ROI Pool Size
    """

    def __init__(self, n_class, spatial_scale, classifier, roi_size = 7):
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.roi_pool = RoIPool( (roi_size, roi_size), spatial_scale)
        self.pred_locs = nn.Linear(4096, n_class * 4)
        self.pred_scores = nn.Linear(4096, n_class)

        init_params(self.pred_locs, 0, 0.001)
        init_params(self.pred_scores, 0, 0.01)


    def forward(self, features, rois, roi_indices):

        """
        Arguments:
            features        :- Image features
            rois            :- A bounding box array containing coordinates of proposal boxes.  
            roi_indices     :- An array of images indices for the bounding boxes.
        Returns:-
            roi_pred_locs   :- The predicted locations from the VGG head for each class
            roi_pred_scores :- The predicted scores from the VGG head for each class
        """
        
        roi_indices = totensor(roi_indices).float()
        rois = totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()

        roi_pool = self.roi_pool(features, indices_and_rois)
        roi_pool = roi_pool.view(roi_pool.size(0), -1)

        classifier_output = self.classifier(roi_pool)

        roi_pred_locs = self.pred_locs(classifier_output)
        roi_pred_scores = self.pred_scores(classifier_output)
        
        return roi_pred_locs, roi_pred_scores

