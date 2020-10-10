import numpy as np

class Anchors(object):

    """
        Contains all the utility functions for operating on anchors
    """

    def __init__(self, anchor_base):
        self.anchor_base = anchor_base

        self.n_sample = 256
        self.pos_iou_thresh = 0.7
        self.neg_iou_thresh = 0.3
        self.pos_ratio = 0.5

    def decode_bbox(self, input_bboxes, offsets_scales):

        """
            Responsible for decoding the input bbox with passed offset and scales values
            Args:-
                input_bboxes   :- An array containing a list of bboxes; top left and bottom right locations.
                offsets_scales :- An array containing the list of bboxes; Y, X centre (offsets) and H, W (scales).
            Returns:-
                output_bboxes  :- An array containing a list of bboxes; top left and bottom right locations.
        """

        input_bboxes = input_bboxes.astype(input_bboxes.dtype, copy=False)
        output_bboxes = np.zeros(offsets_scales.shape, dtype=offsets_scales.dtype)

        input_h  = input_bboxes[:, 2] - input_bboxes[:, 0]
        input_w  = input_bboxes[:, 3] - input_bboxes[:, 1]
        input_cy = input_bboxes[:, 0] + 0.5 * input_h
        input_cx = input_bboxes[:, 1] + 0.5 * input_w

        dy = offsets_scales[:, 0]
        dx = offsets_scales[:, 1]
        dh = offsets_scales[:, 2]
        dw = offsets_scales[:, 3]

        ctr_y = dy * input_h + input_cy
        ctr_x = dx * input_w + input_cx 
        h = np.exp(dh) * input_h
        w = np.exp(dw) * input_w

        output_bboxes[:, 0] = ctr_y - 0.5 * h
        output_bboxes[:, 2] = ctr_y + 0.5 * h
        output_bboxes[:, 1] = ctr_x - 0.5 * w
        output_bboxes[:, 3] = ctr_x + 0.5 * w

        return output_bboxes


    def encode_bbox(self, input_bboxes, output_bboxes):
    
        """
            Responsible for encoding the input bbox with output bbox values
            Args:-
                input_bboxes   :- An array containing a list of bboxes; top left and bottom right locations.
                output_bboxes  :- An array containing a list of bboxes; top left and bottom right locations.
            Returns:-
                offset_scales  :- An array containing the list of bboxes; Y, X centre (offsets) and H, W (scales).
        """

        input_height = input_bboxes[:, 2] - input_bboxes[:, 0]
        input_width = input_bboxes[:, 3] - input_bboxes[:, 1]
        input_cy = input_bboxes[:, 0] + 0.5 * input_height
        input_cx = input_bboxes[:, 1] + 0.5 * input_width

        output_height = output_bboxes[:, 2] - output_bboxes[:, 0]
        output_width = output_bboxes[:, 3] - output_bboxes[:, 1]
        output_cy = output_bboxes[:, 0] + 0.5 * output_height
        output_cx = output_bboxes[:, 1] + 0.5 * output_width

        dy = (output_cy - input_cy) / input_height
        dx = (output_cx - input_cx) / input_width
        dh = np.log(output_height / input_height)
        dw = np.log(output_width / input_width)

        offset_scales = np.vstack((dy, dx, dh, dw)).transpose()
        return offset_scales

    
    def bbox_iou(self, first_bbox, second_bbox):
    
        """
            Calculates the Intersection Over Union of the bboxes
            Args:-
                first_bbox    :- An array containing a list of bboxes; top left and bottom right locations.
                second_bbox   :- An array containing a list of bboxes; top left and bottom right locations.
            Returns:-
                iou           :- An array containing the iou values
        """

        x1 = np.maximum(first_bbox[:, None, 0], second_bbox[:, 0])
        y1 = np.maximum(first_bbox[:, None, 1], second_bbox[:, 1])
        x2 = np.minimum(first_bbox[:, None, 2], second_bbox[:, 2])
        y2 = np.minimum(first_bbox[:, None, 3], second_bbox[:, 3])

        width = (x2 - x1)
        height = (y2 - y1)
        width[width < 0] = 0
        height[height < 0] = 0
        intersection = width * height

        first_bbox_area = np.prod(first_bbox[:, 2:] - first_bbox[:, :2], axis=1)
        second_bbox_area = np.prod(second_bbox[:, 2:] - second_bbox[:, :2], axis=1)
        union = first_bbox_area[:, None] + second_bbox_area - intersection

        iou = intersection / union
        return iou


    def generate_image_anchors(self, height, width, stride):
        
        """
            Generates all the anchors for image from base anchor
            Args:-
                height        :- The height of the image
                width         :- The width of the image
                stride        :- The stride of the image
            Returns:-
                image_anchors :- An array containing all the generated anchors for the image
        """

        w = np.arange(0, width * stride, stride)
        h = np.arange(0, height * stride, stride)
        W, H = np.meshgrid(w, h)
        # shift defines the amount the top left and bottom right for each base anchor is shifted
        shift = np.stack((H.ravel(), W.ravel(), H.ravel(), W.ravel()), axis=1)

        image_anchors = self.anchor_base.reshape((1, -1, 4)) + shift.reshape((1, -1, 4)).transpose((1, 0, 2))
        image_anchors = image_anchors.reshape((-1, 4)).astype(np.float32)

        return image_anchors


    def create_anchor_targets(self, bbox, image_anchors, img_size):

        """
            Computes the two real locations and scores for computing loss with RPN
            Args:-
                bbox           :- The GT bounding boxes of the image
                image_anchors  :- Contains all the possible anchors for the image.
                img_size       :- The size of the image H X W
            Returns:-
                real_locs      :- This is the offsets and scale x, y, h, w for the image anchors.
                real_scores    :- This is the confidence score for the image anchors.
        """

        height, width = img_size
        n_anchor = len(image_anchors)
        inside_index = np.where(
            (image_anchors[:, 0] >= 0) &
            (image_anchors[:, 1] >= 0) &
            (image_anchors[:, 2] <= height) &
            (image_anchors[:, 3] <= width)
        )[0]
        image_anchors = image_anchors[inside_index]

        argmax_ious, real_scores = self.create_anchor_labels(inside_index, image_anchors, bbox)

        # Bounding Box targets needed for RPN regression loss calculation 
        real_locs = self.encode_bbox(image_anchors, bbox[argmax_ious])

        # map up to original set of anchors
        real_scores = _unmap(real_scores, n_anchor, inside_index, fill=-1)
        real_locs = _unmap(real_locs, n_anchor, inside_index, fill=0)

        return real_locs, real_scores


    def create_anchor_labels(self, inside_index, image_anchors, bbox):

        """
            Computes the two real locations and scores for computing loss with RPN
            Args:-
                inside_index   :- The indexes for the bboxes which lie inside image
                image_anchors  :- Contains all the possible anchors for the image.
                bbox           :- The size of the image H X W
            Returns:-
                real_locs      :- This is the offsets and scale x, y, h, w for the image anchors.
                real_scores    :- This is the confidence score for the image anchors.
        """

        real_scores = np.empty((len(inside_index),), dtype=np.int32)
        real_scores.fill(-1)

        ious = self.bbox_iou(image_anchors, bbox)

        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        
        argmax_ious_for_each_bbox = ious.argmax(axis=0)
        max_ious_each_bbox = ious[argmax_ious_for_each_bbox, np.arange(ious.shape[1])]
        argmax_ious_for_each_bbox = np.where(ious == max_ious_each_bbox)[0]

        real_scores[max_ious < self.neg_iou_thresh] = 0
        #  check this
        real_scores[argmax_ious_for_each_bbox] = 1
        real_scores[max_ious >= self.pos_iou_thresh] = 1

        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(real_scores == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            real_scores[disable_index] = -1

        n_neg = self.n_sample - np.sum(real_scores == 1)
        neg_index = np.where(real_scores == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            real_scores[disable_index] = -1

        return argmax_ious, real_scores


def create_anchor_base(scales = [8, 16, 32], ratios = [0.5, 1, 2], stride = 16):

    """
        Creates the initial anchor base from a given set of scales and ratios.
        The base anchors can be of square and rectangle shapes.
        Args:-
            scales      :- A list of all the possible scales. eg :- [8, 16, 32]
            ratios      :- A list of all the possible ratios. eg :- [0.5, 1, 2]
            stride      :- Factor by which VGGNet has downsampled the image
        Returns:-
            anchor_base :-  An Array containing len(scales) * len(ratios) elements.
                            Each Containing 4 values denoting the top left and bottom right positions of anchor box.
    """

    anchor_base = np.zeros((len(ratios) * len(scales), 4), dtype=np.float32)

    for i in range(len(ratios)):
        for j in range(len(scales)):
            index = i * len(scales) + j

            # Calculating the height and width of the bounding box
            h = stride * scales[j] * ratios[i]
            w = stride * scales[j] * (1. / ratios[i])
            
            # Calculating the anchor base height and width
            anchor_base[index, 0] = stride - h / 2.
            anchor_base[index, 2] = stride + h / 2.
            
            anchor_base[index, 1] = stride - w / 2.
            anchor_base[index, 3] = stride + w / 2.

    return anchor_base        

def _unmap(data, count, index, fill=0):

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret

