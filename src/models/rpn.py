import numpy as np
import torch
from torch import nn
from torchvision.ops import nms
from torch.nn import functional as F

from utils.anchors import Anchors
from utils.helper import init_params

class RegionProposalNetwork(nn.Module):

    """
        This implements the RPN to propose the anchors associated with the images.
        Steps Involved include applying CNN to fetch the locations and scores of anchors.
        Then NMS is done to fetch the filtered rois.
    """

    def __init__(self, anchors, feat_stride, in_channels=512, mid_channels=512):
        super(RegionProposalNetwork, self).__init__()
        self.anchors = anchors
        self.n_anchor = self.anchors.anchor_base.shape[0]
        self.feat_stride = feat_stride
        self.convolution = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.rpn_locs = nn.Conv2d(mid_channels, self.n_anchor * 4, 1, 1, 0)
        self.rpn_scores = nn.Conv2d(mid_channels, self.n_anchor * 2, 1, 1, 0)
        init_params(self.convolution)
        init_params(self.rpn_locs)
        init_params(self.rpn_scores)

        # NMS Params
        self.n_pre_nms = 12000 # Applied before NMS to filter top n_pre_nms
        self.min_size = 16     # Applied to filter height, width lower than min_size
        self.n_post_nms = 2000 # Applied before NMS to filter top n_post_nms
        self.nms_thresh = 0.7  # Applied during NMS


    def forward(self, x, img_size, scale):

        """
            Region Proposal Network Fwd
            Arguments :-
                x             :- The image features (N, C, H, W)
                img_size      :- The size of the image H X W
                scale         :- scaling applied during preproceesing
            Returns :-
                pred_locs     : This is the predicted offsets and scale x, y, h, w for the anchors`.
                pred_scores   : This is the predicted confidence score for the anchors
                rois          : Contains all the possible rois for the images after filtration steps
                roi_indices   : Corresponds to the images which each roi belongs to.
                image_anchors : Contains all the possible anchors for the image.   
        """
        self.scale = scale
        
        n, _, height, width = x.shape
        image_anchors = self.anchors.generate_image_anchors(height, width, self.feat_stride)

        convolution_output = F.relu(self.convolution(x))
        pred_locs = self.rpn_locs(convolution_output)
        pred_locs = pred_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        
        pred_scores = self.rpn_scores(convolution_output)
        pred_scores = pred_scores.permute(0, 2, 3, 1).contiguous()

        pred_softmax_scores = F.softmax(pred_scores.view(n, height, width, self.n_anchor, 2), dim=4)
        pred_scores = pred_scores.view(n, -1, 2)
        # Predicted foreground confidence scores
        pred_fg_softmax_scores = pred_softmax_scores[:, :, :, :, 1].contiguous()
        pred_fg_softmax_scores = pred_fg_softmax_scores.view(n, -1)
        
        rois, roi_indices = self.filer_rois(pred_locs, pred_fg_softmax_scores, image_anchors, img_size, n)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return pred_locs, pred_scores, rois, roi_indices, image_anchors


    def filer_rois(self, pred_locs, pred_fg_softmax_scores, image_anchors, img_size, n):
        rois = []
        roi_indices = []
        for i in range(n):
            roi = self.filer_roi(pred_locs[i].data.numpy(), pred_fg_softmax_scores[i].data.numpy(), image_anchors, img_size)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        return rois, roi_indices


    def filer_roi(self, pred_loc, pred_fg_softmax_score, image_anchors, img_size):

        """
            This applies the NMS to the rois to fetch the filtered results.
            It undergoes 4 layers of filtrations along with clipping.
            Arguments :-
                pred_loc              :- This is the predicted offsets and scale x, y, h, w for the anchors
                pred_fg_softmax_score :- This is the predicted foreground confidence score for the anchors
                image_anchors         :- An array containing the list of possible anchors for the images
                img_size              :- The size of the image H X W
            Returns :-
                roi                   :- Returns the eligible roi after going through all filters
        """
        
        roi = self.anchors.decode_bbox(image_anchors, pred_loc)

        # Clip those boxes which lie outside image dimensions
        roi[:, 0] = np.clip(roi[:, 0], 0, img_size[0])
        roi[:, 2] = np.clip(roi[:, 2], 0, img_size[0])
        roi[:, 1] = np.clip(roi[:, 1], 0, img_size[1])
        roi[:, 3] = np.clip(roi[:, 3], 0, img_size[1])
        roi_height = roi[:, 2] - roi[:, 0]
        roi_width = roi[:, 3] - roi[:, 1]

        # 1. Min Size Filteration 
        min_size = self.min_size * self.scale
        keep = np.where((roi_height >= min_size) & (roi_width >= min_size))[0]
        roi = roi[keep, :]
        pred_fg_softmax_score = pred_fg_softmax_score[keep]
        order = pred_fg_softmax_score.ravel().argsort()[::-1]
        
        # 2. Pre NMS Filteration 
        order = order[:self.n_pre_nms ]
        roi = roi[order, :]
        pred_fg_softmax_score = pred_fg_softmax_score[order]

        # 3. NMS Filteration 
        keep = nms(torch.from_numpy(roi), torch.from_numpy(pred_fg_softmax_score), self.nms_thresh)
        
        # 4. Post NMS Filteration 
        keep = keep[:self.n_post_nms]
        
        roi = roi[keep.cpu().numpy()]
        return roi