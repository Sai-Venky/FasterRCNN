from __future__ import  absolute_import
import os
import time
from collections import namedtuple

import numpy as np
import torch as t
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from torchnet.meter import ConfusionMeter, AverageValueMeter

from models.head import get_rcnn_vgg16, VGG16RoIHead
from models.rpn import RegionProposalNetwork

from utils.config import opt
from utils.helper import init_params, tonumpy, totensor, scalar, convert_cuda
from utils.visualization import Visualizer
from utils.anchors import Anchors, create_anchor_base
from utils.proposals import ProposalTargetCreator
from data.util import preprocess

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])

def nograd(f):
    def new_f(*args,**kwargs):
        with t.no_grad():
           return f(*args,**kwargs)
    return new_f

class FasterRCNN(nn.Module):
    
    """
        Implementation of FasterRCNN Architecture
    """

    def __init__(self):
        super(FasterRCNN, self).__init__()

        # Feature Extracter
        self.extractor, self.classifier = get_rcnn_vgg16()
        
        # RPN 
        self.feat_stride = 16
        self.anchor_base = create_anchor_base(self.feat_stride)
        self.anchors = Anchors(self.anchor_base)
        self.rpn = RegionProposalNetwork(self.anchors, self.feat_stride)

        self.loc_normalize_mean=(0., 0., 0., 0.)
        self.loc_normalize_std=(0.1, 0.1, 0.2, 0.2)
        self.proposal_target_creator = ProposalTargetCreator(self.anchors, self.loc_normalize_mean, self.loc_normalize_std)

        # VGGHead   
        self.n_class = 21
        self.head = VGG16RoIHead(n_class=self.n_class, spatial_scale=(1. / self.feat_stride), classifier=self.classifier)
        
        self.vis = Visualizer(env=opt.env)
        self.optimizer = self.get_optimizer()
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # Indicators for Training
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # Average Loss
        self.cuda = t.cuda.is_available()

        # Prediction
        self.pred_score_thresh = 0.7
        self.pred_nms_thresh = 0.7

    def forward(self, imgs, bboxes, labels, scale):
        
        """
        Forward Faster R-CNN 

        Arguments:
            imgs        :- A variable with a batch of images.
            bboxes      :- A batch of bounding boxes.
            labels      :- A batch of labels.
            scale       :- scaling applied during preprocessing

        Returns:
            LossTuple   :- namedtuple of all the five losses
        """

        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.extractor(imgs)

        pred_locs, pred_scores, rois, roi_indices, image_anchors = self.rpn(features, img_size, scale)

        bbox = bboxes[0]
        label = labels[0]
        rpn_score = pred_scores[0]
        rpn_loc = pred_locs[0]
        roi = rois

        selected_rois, roi_bbox_iou, keep_index = self.proposal_target_creator.generate_proposals(roi, tonumpy(bbox))

        selected_rois_index = t.zeros(len(selected_rois))
        roi_cls_loc, roi_score = self.head(
            features,
            selected_rois,
            selected_rois_index)

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchors.create_anchor_targets(
            tonumpy(bbox),
            image_anchors,
            img_size)
        gt_rpn_label = totensor(gt_rpn_label).long()
        gt_rpn_loc = totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        rpn_cls_loss = F.cross_entropy(rpn_score, convert_cuda(gt_rpn_label, self.cuda), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = tonumpy(rpn_score)[tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses -------------------#
        gt_roi_loc, gt_roi_label = self.proposal_target_creator.generate_roi_gt_values(selected_rois, roi_bbox_iou, tonumpy(label), tonumpy(bbox), keep_index)

        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[ convert_cuda(t.arange(0, n_sample).long(), self.cuda), \
                              totensor(gt_roi_label).long()]
        gt_roi_label = totensor(gt_roi_label).long()
        gt_roi_loc = totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, convert_cuda(gt_roi_label, self.cuda))

        self.roi_cm.add(totensor(roi_score, False), gt_roi_label.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses

    @nograd
    def predict(self, imgs, scale):
        self.eval()

        bboxes = list()
        labels = list()
        scores = list()

        for img in imgs:
            img = totensor(img[None]).float()
            _, _, H, W = img.shape
            img_size = (H, W)

            features = self.extractor(img)
            pred_locs, pred_scores, rois, roi_indices, image_anchors = self.rpn(features, img_size, scale)
            roi_cls_loc, roi_score = self.head(
                features,
                rois,
                roi_indices)            
        
            roi_score = roi_score.data
            roi_cls_loc = roi_cls_loc.data
            roi = totensor(rois)

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = t.Tensor(self.loc_normalize_mean). \
                repeat(self.n_class)[None]
            std = t.Tensor(self.loc_normalize_std). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = self.anchors.decode_bbox(tonumpy(roi).reshape((-1, 4)),
                                tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=img_size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=img_size[1])

            prob = (F.softmax(totensor(roi_score), dim=1))

            bbox, label, score = self.select_boxes(cls_bbox, prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.train()
        return bboxes, labels, scores

    def select_boxes(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()

        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.pred_score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l, self.pred_nms_thresh)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())

        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score


    def update_meters(self, losses):
        loss_d = {k: scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()


    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


    def get_optimizer(self):
        
        """
            returns Optimizer for FasterRCNN
        """
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': opt.lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': opt.lr, 'weight_decay': opt.weight_decay}]
        self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer


    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer


    def save(self):
        timestr = time.strftime('%m%d%H%M')
        save_path = 'checkpoints/fasterrcnn_%s' % timestr

        save_dict = dict()
        save_dict['model'] = self.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['optimizer'] = self.optimizer.state_dict()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        return save_path


    def load(self, path):
        state_dict = t.load(path, map_location=t.device('cpu'))
        self.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        opt._parse(state_dict['config'])
        return self

def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape)

    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    loc_loss /= ((gt_label >= 0).sum().float())
    return loc_loss
