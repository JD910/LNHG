from __future__ import absolute_import

import os
import time
import numpy as np
from collections import namedtuple

import torch as torch
from torch import nn
from torch.nn import functional as F

from utils.utils import AnchorTargetCreator, ProposalTargetCreator

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])

class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn,optimizer):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = 1
        self.roi_sigma = 1

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = [0, 0, 0, 0]
        self.loc_normalize_std = [0.1, 0.1, 0.2, 0.2]

        self.optimizer = optimizer

    def forward(self, imgs, bboxes, labels, scale):
        n = imgs.shape[0]
        img_size = imgs.shape[2:]

        base_feature = self.faster_rcnn.extractor(imgs)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(base_feature, img_size, scale)

        rpn_loc_loss_all, rpn_cls_loss_all, roi_loc_loss_all, roi_cls_loss_all = 0, 0, 0, 0
        for i in range(n):
            bbox = bboxes[i]
            label = labels[i]
            rpn_loc = rpn_locs[i]
            rpn_score = rpn_scores[i]
            roi = rois[roi_indices==i]
            feature = base_feature[i]

            gt_rpn_loc, gt_rpn_label,get_pre_anchor = self.anchor_target_creator(bbox, anchor, img_size)
            if(np.any(get_pre_anchor == 0)):
                continue
            if(np.any(bbox == 0) | (bbox.size == 0)):
                continue
            gt_rpn_loc = torch.Tensor(gt_rpn_loc)
            gt_rpn_label = torch.Tensor(gt_rpn_label).long()
            get_pre_anchor = torch.Tensor(get_pre_anchor)
            get_bbox = torch.Tensor(bbox)
            
            if rpn_loc.is_cuda:
                gt_rpn_loc = gt_rpn_loc.cuda()
                gt_rpn_label = gt_rpn_label.cuda()
                get_pre_anchor = get_pre_anchor.cuda()
                get_bbox = get_bbox.cuda()

            rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, get_bbox, get_pre_anchor, self.rpn_sigma)
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)
  
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label, self.loc_normalize_mean, self.loc_normalize_std)
            sample_roi = torch.Tensor(sample_roi)
            gt_roi_loc = torch.Tensor(gt_roi_loc)
            gt_roi_label = torch.Tensor(gt_roi_label).long()
            sample_roi_index = torch.zeros(len(sample_roi))
            
            if feature.is_cuda:
                sample_roi = sample_roi.cuda()
                sample_roi_index = sample_roi_index.cuda()
                gt_roi_loc = gt_roi_loc.cuda()
                gt_roi_label = gt_roi_label.cuda()

            roi_cls_loc, roi_score = self.faster_rcnn.head(torch.unsqueeze(feature, 0), sample_roi, sample_roi_index, img_size)

            n_sample = roi_cls_loc.size()[1]
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, n_sample), gt_roi_label]


            roi_loc_loss = _fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label.data,get_bbox, get_pre_anchor, self.roi_sigma)
            roi_cls_loss = nn.CrossEntropyLoss()(roi_score[0], gt_roi_label)

            rpn_loc_loss_all += rpn_loc_loss.mean()
            rpn_cls_loss_all += rpn_cls_loss
            roi_loc_loss_all += roi_loc_loss.mean()
            roi_cls_loss_all += roi_cls_loss
            
        losses = [rpn_loc_loss_all/n, rpn_cls_loss_all/n, roi_loc_loss_all/n, roi_cls_loss_all/n]
        losses = losses + [sum(losses)]
        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        return losses

def _smooth_l1_loss(pred_loc, gt_loc, bbox, get_pre_anchor, sigma, eps = 1e-7):

    '''
    sigma_squared = sigma ** 2
    regression_diff = (pred_loc - gt_loc)
    regression_diff = regression_diff.abs()
    regression_loss = torch.where(
             regression_diff < (1. / sigma_squared),
             0.5 * sigma_squared * regression_diff ** 2,
             regression_diff - 0.5 / sigma_squared
         )
    return regression_loss.sum()
    '''


    '''
    https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py#L36
    :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :return: GIoU loss
    '''
    ix1 = torch.max(pred_loc[:, 0], gt_loc[:, 0])
    iy1 = torch.max(pred_loc[:, 1], gt_loc[:, 1])
    ix2 = torch.min(pred_loc[:, 2], gt_loc[:, 2])
    iy2 = torch.min(pred_loc[:, 3], gt_loc[:, 3])

    iw = (ix2 - ix1 + 1.0).clamp(0.)
    ih = (iy2 - iy1 + 1.0).clamp(0.)

    # overlap
    inters = iw * ih

    # union
    uni = (pred_loc[:, 2] - pred_loc[:, 0] + 1.0) * (pred_loc[:, 3] - pred_loc[:, 1] + 1.0) + (gt_loc[:, 2] - gt_loc[:, 0] + 1.0) * (
            gt_loc[:, 3] - gt_loc[:, 1] + 1.0) - inters + eps

    # ious
    ious = inters / uni

    ex1 = torch.min(pred_loc[:, 0], gt_loc[:, 0])
    ey1 = torch.min(pred_loc[:, 1], gt_loc[:, 1])
    ex2 = torch.max(pred_loc[:, 2], gt_loc[:, 2])
    ey2 = torch.max(pred_loc[:, 3], gt_loc[:, 3])
    ew = (ex2 - ex1 + 1.0).clamp(min=0.)
    eh = (ey2 - ey1 + 1.0).clamp(min=0.)

    # enclose erea
    enclose = ew * eh + eps

    giou = ious - (enclose - uni) / enclose

    loss = 1 - giou

    loss = torch.sum(loss)

    return loss

def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, bbox, get_pre_anchor, sigma):
    pred_loc = pred_loc[gt_label>0]
    gt_loc = gt_loc[gt_label>0]

    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, bbox, get_pre_anchor, sigma)
    num_pos = (gt_label > 0).sum().float()
    loc_loss /= torch.max(num_pos, torch.ones_like(num_pos))
    return loc_loss
