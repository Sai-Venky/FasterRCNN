import os
import xml.etree.ElementTree as ET
import numpy as np
from .util import read_image

class VOCDataset:

    """
    Bounding box dataset VOC.

    The bounding boxes are in shape (R, 4), where R is the number of bounding boxes 
    The 4 values represent :- `(y_{min}, x_{min}, y_{max}, x_{max})`,
    which are the coordinates of the top left and the bottom right vertices.

    The labels are packed into a one dimensional tensor of shape (R,).
    The class name of the label are the elements of VOC_BBOX_LABEL_NAMES.

    Arguments:
        data_dir      :- Path to the root of the training data. 
        split         :- {'train', 'val', 'trainval', 'test'}

    """

    def __init__(self, data_dir, split='trainval'):

        id_list_file = os.path.join(data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        
        """
        Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Arguments:
            i (int) :- The index of the example.
        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]

        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')

        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):

            if int(obj.find('difficult').text) == 1:
                continue
            difficult.append(int(obj.find('difficult').text))

            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])

            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))

        img = read_image(img_file)
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)

        return img, bbox, label, difficult

    __getitem__ = get_example


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
)