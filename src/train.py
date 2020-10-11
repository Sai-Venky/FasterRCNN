from __future__ import  absolute_import
import os
import ipdb
import matplotlib
from tqdm import tqdm
from torch.utils import data as data_
from data.dataset import Dataset, inverse_normalize
from models.faster_rcnn import FasterRCNN
from utils.helper import tonumpy, totensor, scalar
from utils.visualization import visdom_bbox
from utils.config import opt

matplotlib.use('agg')


def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)

    print('Loading fom the Dataset')
    dataloader = data_.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.num_workers)
    faster_rcnn = FasterRCNN()
    print('Model Construct Completed')

    if opt.load_path:
        faster_rcnn.load(opt.load_path)
        print('Load Pretrained Model from %s' % opt.load_path)

    faster_rcnn.vis.text(dataset.voc.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr

    for epoch in range(opt.epoch):
        faster_rcnn.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):

            scale = scalar(scale)
            img, bbox, label = img.float(), bbox_, label_
            losses = faster_rcnn.train_step(img, bbox, label, scale)

            if (ii + 1) % opt.plot_every == 0:

                # plot loss
                faster_rcnn.vis.plot_many(faster_rcnn.get_meter_data())

                ori_img_ = inverse_normalize(tonumpy(img[0]))
                # plot Groudtruth bboxes
                gt_img = visdom_bbox(ori_img_,
                                     tonumpy(bbox_[0]),
                                     tonumpy(label_[0]))
                faster_rcnn.vis.img('gt_img', gt_img)

                # plot Predicted bboxes
                _bboxes, _labels, _scores = faster_rcnn.predict([ori_img_], scale)
                pred_img = visdom_bbox(ori_img_,
                                       tonumpy(_bboxes[0]),
                                       tonumpy(_labels[0]).reshape(-1),
                                       tonumpy(_scores[0]))
                faster_rcnn.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                faster_rcnn.vis.text(str(faster_rcnn.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                faster_rcnn.vis.img('roi_cm', totensor(faster_rcnn.roi_cm.conf, False).float())

        lr_ = faster_rcnn.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, loss:{}'.format(str(lr_), str(faster_rcnn.get_meter_data()))
        faster_rcnn.vis.log(log_info)

        if epoch == 9:
            faster_rcnn.load(best_path)
            faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13: 
            break


if __name__ == '__main__':
    train()