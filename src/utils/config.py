from pprint import pprint

class Config:
    voc_data_dir = '/Users/ecom-v.ramesh/Documents/Personal/2020/DL/simple-faster-rcnn-pytorch/VOCdevkit/VOC2007/'

    num_workers = 0

    # Sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # Params for Optimizer
    weight_decay = 0.0005
    lr_decay = 0.1
    lr = 1e-3

    # Visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plot_every = 1

    # Training
    epoch = 14

    load_path = None

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
