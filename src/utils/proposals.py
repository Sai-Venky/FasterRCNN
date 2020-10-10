from utils.anchors import Anchors
from utils.helper import init_params
import numpy as np
import torch as t
from torch import nn
from torch.nn import functional as F

class ProposalTargetCreator(object):

    """Assign ground truth bounding boxes to given RoIs.

    Args:
        n_sample (int): The number of sampled regions.
        pos_ratio (float): Fraction of regions that is labeled as a
            foreground.
        pos_iou_thresh (float): IoU threshold for a RoI to be considered as a
            foreground.
        neg_iou_thresh_hi (float): RoI is considered to be the background
            if IoU is in
            [:obj:`neg_iou_thresh_hi`, :obj:`neg_iou_thresh_hi`).
        neg_iou_thresh_lo (float): See above.

    """

    def __init__(self, anchors):

        self.anchors = anchors

        # Criterias for proposal selections
        self.n_sample = 128
        self.pos_ratio = 0.25
        self.pos_iou_thresh = 0.5
        self.neg_iou_thresh_hi = 0.5
        self.neg_iou_thresh_lo = 0.0


    def generate_proposals(self, roi, bbox):

        n_bbox, _ = bbox.shape

        # check this
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
        roi = roi[keep_index]

        return roi, roi_bbox_iou, keep_index

    
    def generate_roi_gt_values(self, selected_rois, roi_bbox_iou, labels, bbox, keep_index, loc_normalize_mean=(0., 0., 0., 0.), loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):

        """
            Computes the two real locations and scores for computing loss with ROI

        Args:
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
        real_locs = ((real_locs - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))
        return real_locs, real_scores