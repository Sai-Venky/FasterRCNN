import numpy as np
import torch as t
from torch import nn
from torch.nn import functional as F

from utils.anchors import Anchors
from utils.helper import init_params

class ProposalTargetCreator(object):

    """
        Generated the proposals for the final VGGNet Head.
        Assign ground truth bounding boxes to given RoIs.
    """

    def __init__(self, anchors, loc_normalize_mean, loc_normalize_std):

        self.anchors = anchors

        # Criterias for proposal selections
        self.n_sample = 128
        self.pos_ratio = 0.25
        self.pos_iou_thresh = 0.5
        self.neg_iou_thresh_hi = 0.5
        self.neg_iou_thresh_lo = 0.0

        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std


    def generate_proposals(self, roi, bbox):

        """
            Generates the final proposals
            Arguments:
                roi            :- The ROIs from which proposals are to be selected from.
                bbox           :- The GT bounding boxes of the image
            Returns:-
                selected_rois  :- The ROIs selected as proposals
                roi_bbox_iou   :- The iou between the bboxes and rois
                keep_index     :- The indexes chosen as proposals
        """

        n_bbox, _ = bbox.shape

        roi = np.concatenate((roi, bbox), axis=0)

        roi_bbox_iou = self.anchors.bbox_iou(roi, bbox)
        max_iou = roi_bbox_iou.max(axis=1)

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]

        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        self.pos_roi_per_this_image = pos_roi_per_this_image
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        selected_rois = roi[keep_index]

        return selected_rois, roi_bbox_iou, keep_index

    
    def generate_roi_gt_values(self, selected_rois, roi_bbox_iou, labels, bbox, keep_index):

        """
            Computes the real locations and scores for calculating the loss with ROI
            Arguments:
                selected_rois  :- The ROIs selected as proposals
                roi_bbox_iou   :- The iou between the bboxes and rois
                labels         :- The ground truth labels of the bounding boxes
                bbox           :- The GT bounding boxes of the image
            Returns:-
                real_locs      :- This is the offsets and scale x, y, h, w for the image anchors.
                real_scores    :- This is the confidence score for the image anchors.
        """

        argmax_ious = roi_bbox_iou.argmax(axis=1)

        real_scores = labels[argmax_ious] + 1        
        real_scores = real_scores[keep_index]
        real_scores[self.pos_roi_per_this_image:] = 0

        real_locs = self.anchors.encode_bbox(selected_rois, bbox[argmax_ious[keep_index]])
        real_locs = ((real_locs - np.array(self.loc_normalize_mean, np.float32)
                       ) / np.array(self.loc_normalize_std, np.float32))
        return real_locs, real_scores