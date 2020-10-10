from __future__ import  absolute_import
from utils.anchors import Anchors, create_anchor_base
from utils.helper import init_params, totensor
from utils.proposals import ProposalTargetCreator
from models.head import get_rcnn_vgg16, VGG16RoIHead
from models.rpn import RegionProposalNetwork
import numpy as np
import torch as t
from utils.helper import tonumpy,totensor,scalar
from torch import nn
from torchvision.ops import nms
from torch.nn import functional as F
import os
from collections import namedtuple
import time
from data.util import preprocess

from utils.visualization import Visualizer

from utils.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter

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
        Implementation of FasterRCNN
    """

    def __init__(self):
        super(FasterRCNN, self).__init__()
        
        self.feat_stride = 16
        self.extractor, self.classifier = get_rcnn_vgg16()
        self.anchor_base = create_anchor_base()
        self.anchors = Anchors(self.anchor_base)
        self.rpn = RegionProposalNetwork(self.anchors, self.feat_stride)
        self.proposal_target_creator = ProposalTargetCreator(self.anchors)
        # self.head = VGG16RoIHead()
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma
        self.head = VGG16RoIHead(
            n_class=21,
            spatial_scale=(1. / self.feat_stride),
            classifier=self.classifier)

        self.optimizer = self.get_optimizer()
        # visdom wrapper
        self.vis = Visualizer(env=opt.env)

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def forward(self, imgs, bboxes, labels, scale):
        """
        Forward Faster R-CNN 

        Args:
            imgs :- A variable with a batch of images.
            bboxes :- A batch of bounding boxes.
            labels :- A batch of labels.

        Returns:
            namedtuple of 5 losses
        """
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.extractor(imgs)

        pred_locs, pred_scores, rois, roi_indices, image_anchors = self.rpn(features, img_size)

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
        gt_roi_loc, gt_roi_label = self.proposal_target_creator.generate_roi_gt_values(selected_rois, roi_bbox_iou, tonumpy(label), tonumpy(bbox), keep_index)
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

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = tonumpy(rpn_score)[tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long(), \
                              totensor(gt_roi_label).long()]
        gt_roi_label = totensor(gt_roi_label).long()
        gt_roi_loc = totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)

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
    def predict(self, imgs):
        self.eval()
        prepared_imgs = list()
        sizes = list()                  
        self.use_preset('visualize')
        for img in imgs:
            size = img.shape[1:]
            prepared_imgs.append(img)
            sizes.append(size)

        bboxes = list()
        labels = list()
        scores = list()

        for img, size in zip(prepared_imgs, sizes):
            img = totensor(img[None]).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)

            img_size = x.shape[2:]

            features = self.extractor(img)
            pred_locs, pred_scores, rois, roi_indices, image_anchors = self.rpn(features, img_size)
            roi_cls_loc, roi_score = self.head(
                features,
                rois,
                roi_indices)            
        
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = totensor(rois) / scale

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
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = (F.softmax(totensor(roi_score), dim=1))

            bbox, label, score = self._suppress(cls_bbox, prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l,self.nms_thresh)
            # import ipdb;ipdb.set_trace()
            # keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score


    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else: 
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

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
        return optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer

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
