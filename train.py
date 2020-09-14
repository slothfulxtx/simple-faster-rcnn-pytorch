from __future__ import absolute_import
import os

import ipdb
import matplotlib
import torch
from tqdm import tqdm

from utils.config import opt
from data.dataset import TrainDataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16, FasterRCNNResNet101
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox, vis_dict
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [
                                                                       sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num:
            break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt.parse(kwargs)

    print('loading data...')

    trainset = TrainDataset(opt)
    train_dataloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = torch.utils.data.DataLoader(
        testset, batch_size=1, num_workers=opt.test_num_workers, shuffle=False, pin_memory=True)

    print('constructing model...')

    if opt.model == 'vgg16':
        faster_rcnn = FasterRCNNVGG16()
    elif opt.model == 'resnet101':
        faster_rcnn = FasterRCNNResNet101()

    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    print('loading model...')

    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    else:
        print('no pretrained model found')

    trainer.vis.text('<br/>'.join(trainset.db.label_names), win='labels')

    print('start training...')

    best_map = 0.0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        print("epoch : %d training ..." % epoch)
        trainer.reset_meters()
        for ii, (imgs_, bboxes_, labels_, scales_) in tqdm(enumerate(train_dataloader)):
            scales = at.scalar(scales_)
            imgs, bboxes, labels = imgs_.cuda().float(), bboxes_.cuda(), labels_.cuda()
            trainer.train_step(imgs, bboxes, labels, scales)

            if (ii + 1) % opt.plot_every == 0:

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # generate plotted image

                img = inverse_normalize(at.tonumpy(imgs_[0]))

                # plot groud truth bboxes
                bbox = at.tonumpy(bboxes_[0])
                label = at.tonumpy(labels_[0])
                img_gt = visdom_bbox(img, bbox, label)
                trainer.vis.img('ground truth', img_gt)

                bboxes__, labels__, scores__ = trainer.faster_rcnn.predict(
                    [img], visualize=True)

                # plot prediction bboxes
                bbox = at.tonumpy(bboxes__[0])
                label = at.tonumpy(labels__[0]).reshape(-1)
                score = at.tonumpy(scores__[0])
                img_pred = visdom_bbox(img, bbox, label, score)
                trainer.vis.img('prediction', img_pred)

                # rpn confusion matrix(meter)
                trainer.vis.text(
                    str(trainer.rpn_cm.value().tolist()), win='rpn_cm')

                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(
                    trainer.roi_cm.conf, False).float())

            if ii + 1 == opt.train_num:
                break

        print("epoch : %d evaluating ..." % epoch)

        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = vis_dict({
            'epoch': '%s/%s' % (str(epoch), str(opt.epoch)),
            'lr': lr_,
            'map': float(eval_result['map']),
        }, trainer.losses_data())

        trainer.vis.log(log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map="%.4f" % best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay


if __name__ == '__main__':
    import fire
    # 使用fire库将程序的全部内容暴露给命令行
    fire.Fire()
