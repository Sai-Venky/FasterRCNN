import time

import numpy as np
import matplotlib
import torch as t
import visdom

matplotlib.use('Agg')
from matplotlib import pyplot as plot

VOC_BBOX_LABEL_NAMES = (
    'fly',
    'bike',
    'bird',
    'boat',
    'pin',
    'bus',
    'c',
    'cat',
    'chair',
    'cow',
    'table',
    'dog',
    'horse',
    'moto',
    'p',
    'plant',
    'shep',
    'sofa',
    'train',
    'tv',
)


def vis_image(img, ax=None):

    """
    Visualize a color image.
    Arguments:
        img  :- A rgb value image
        ax   :- The visualization is displayed on this
    Returns:
        Returns the Axes object with the plot
    """

    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)
    img = img.transpose((1, 2, 0))
    ax.imshow(img.astype(np.uint8))
    return ax


def vis_bbox(img, bbox, label=None, score=None, ax=None):

    """
    Visualize bounding boxes inside image.
    Arguments:
        imgs        :- A variable with a batch of images.
        bboxes      :- A batch of bounding boxes.
        labels      :- A batch of labels.
        score       :- value indicating how confident the prediction is.
        label_names :- Name of labels ordered according to label ids.
        ax          :- The visualization is displayed on this axis
    Returns:
        Returns the Axes object with the plot
    """

    label_names = list(VOC_BBOX_LABEL_NAMES) + ['bg']
    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    ax = vis_image(img, ax=ax)

    if len(bbox) == 0:
        return ax

    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(plot.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=2))

        caption = list()

        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax


def fig2data(fig):

    """
    Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    Arguments:
        fig :- a matplotlib figure
    Returns:
        Returns a numpy 3D array of RGBA values
    """

    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    buf = np.roll(buf, 3, axis=2)
    return buf.reshape(h, w, 4)


def fig4vis(fig):

    """
    convert figure to ndarray
    """

    ax = fig.get_figure()
    img_data = fig2data(ax).astype(np.int32)
    plot.close()
    # HWC->CHW
    return img_data[:, :, :3].transpose((2, 0, 1)) / 255.


def visdom_bbox(*args, **kwargs):
    fig = vis_bbox(*args, **kwargs)
    data = fig4vis(fig)
    return data


class Visualizer(object):

    """
    wrapper for visdom
    you can still access naive visdom function by 
    self.line, self.scater,self._send,etc.
    due to the implementation of `__getattr__`
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom('localhost',env=env, use_incoming_socket=False, **kwargs)
        self._vis_kw = kwargs

        # e.g.('loss',23) the 23th value of loss
        self.index = {}
        self.log_text = ''


    def reinit(self, env='default', **kwargs):

        """
        change the config of visdom
        """

        self.vis = visdom.Visdom(env=env, **kwargs)
        return self


    def plot_many(self, d):

        """
        plot multi values
        @params d: dict (name,value) i.e. ('loss',0.11)
        """

        for k, v in d.items():
            if v is not None:
                self.plot(k, v)


    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)


    def plot(self, name, y, **kwargs):

        """
        self.plot('loss',1.00)
        """

        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1


    def img(self, name, img_, **kwargs):

        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        !!don't ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~!!
        """

        self.vis.images(t.Tensor(img_).cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )


    def log(self, info, win='log_text'):

        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), \
            info=info))
        self.vis.text(self.log_text, win)


    def __getattr__(self, name):
        return getattr(self.vis, name)


    def state_dict(self):
        return {
            'index': self.index,
            'vis_kw': self._vis_kw,
            'log_text': self.log_text,
            'env': self.vis.env
        }


    def load_state_dict(self, d):
        self.vis = visdom.Visdom(env=d.get('env', self.vis.env), **(self.d.get('vis_kw')))
        self.log_text = d.get('log_text', '')
        self.index = d.get('index', dict())
        return self
