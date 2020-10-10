import numpy as np
from PIL import Image
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
import torch as t

def read_image(path, dtype=np.float32):

    """
    This function reads an image from given path.
    Args:
        path           :- A path of image file.
        dtype          :- The type of array. The default value is :obj:`~numpy.float32`.
    Returns:
        ~numpy.ndarray :- An image.
    """

    f = Image.open(path)
    try:
        img = f.convert('RGB')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def resize_bbox(bbox, in_size, out_size):

    """
    Resizes Bounding Boxes

    Args:
        bbox            :- The GT bounding boxes of the image
        in_size         :- A tuple with height and the width of the image before resized.
        out_size        :- A tuple with height and the width of the image after resized.
    Returns:
        ~numpy.ndarray  :- Bounding boxes rescaled according to the given image shapes.
    """

    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def flip_bbox(bbox, size, y_flip=False, x_flip=False):

    """
    Flips Bounding Boxes.

    Args:
        bbox            :- The GT bounding boxes of the image
        size            :- A tuple with height and the width of the image before resized.
        y_flip (bool)   :- vertical flip of an image.
        x_flip (bool)   :- horizontal flip ofan image.
    Returns:
        ~numpy.ndarray  :- Bounding boxes flipped according to the given image shapes.
    """

    H, W = size
    bbox = bbox.copy()

    y_flip, x_flip = False, False
    
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox


def transform(in_data):

    """
    Transforms the image

    Args:
        in_data :- Contains the image, bboxes and labels 
    Returns:
        img     :- The images in RGB format
        bbox    :- The bboxes of the images
        label   :- The labels of the images
        scale   :- The scale factor
        
    """
    
    img, bbox, label = in_data
    _, H, W = img.shape

    img = preprocess(img)

    _, o_H, o_W = img.shape
    bbox = resize_bbox(bbox, (H, W), (o_H, o_W))
    bbox = flip_bbox(bbox, (o_H, o_W), x_flip=True)
    
    scale = o_H / H

    return img, bbox, label, scale   


def preprocess(img, min_size=600, max_size=1000):
    
    """
    Preprocess an image for feature extraction.

    Args:
        img :- The images in RGB format
    Returns:
        A preprocessed image.
    """

    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    return pytorch_normalze(img)    


def pytorch_normalze(img):
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()    