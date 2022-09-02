#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import math
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):  # pre是当前batch所有正样本预测框,target使其对应的标签
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )  # 得到相交区域的左上角点
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )  # 得到相交区域的右下角点。

        area_p = torch.prod(pred[:, 2:], 1)  # 预测框的面积
        area_g = torch.prod(target[:, 2:], 1)  # gt框的面积

        en = (tl < br).type(tl.type()).prod(dim=1)  # 检查是否br都大于tl
        area_i = torch.prod(br - tl, 1) * en  # 相交区域面积
        area_u = area_p + area_g - area_i  # 相并区域面积
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class YOLOLoss(nn.Module):
    def __init__(self, num_classes, fp16, strides=[8, 16, 32], focal_loss=False):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        # nn.BCEWithLogitsLoss
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")  # none时 返回每个正样本和预测框的iou_loss矩阵，也可以指定mean或sum对所有loss求平均或者求和
        self.grids = [torch.zeros(1)] * len(strides)  # [0,0,0]
        self.fp16 = fp16
        self.focal_loss = focal_loss
        self.alpha = 0.25
        self.gamma = 2
        self.focal_loss_ratio = 5

    def forward(self, inputs, labels=None):
        outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        # -----------------------------------------------#
        # inputs    [[batch_size, num_classes + 5, 20, 20]
        #            [batch_size, num_classes + 5, 40, 40]
        #            [batch_size, num_classes + 5, 80, 80]]
        # outputs   [[batch_size, 400, num_classes + 5]
        #            [batch_size, 1600, num_classes + 5]
        #            [batch_size, 6400, num_classes + 5]]
        # x_shifts  [[batch_size, 400]
        #            [batch_size, 1600]
        #            [batch_size, 6400]]
        # -----------------------------------------------#
        for k, (stride, output) in enumerate(zip(self.strides, inputs)):
            output, grid = self.get_output_and_grid(output, k, stride)
            x_shifts.append(grid[:, :, 0])  # 每一个元素代表不同特征层网格的x坐标点。
            y_shifts.append(grid[:, :, 1])  # 代表不同特征层网格的有y坐标点
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)  # 代表不同特征层的步距 shape=[1,6400]
            outputs.append(output)

        return self.get_losses(x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1), self.focal_loss)

    # 利用预测参数对网格点进行调整，然后乘上步距，得到原图的xywh,也就说此时output存储了预测框在原图上的xywh，grid是网格点
    def get_output_and_grid(self, output, k, stride):  # output是网络其中一层的输出
        grid = self.grids[k]
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])  # 生成网格点，yv表示网格的纵坐标，xv表示网格的横坐标
            grid = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(output.type())  # 生成网格左上角点 shape=[1,80,80,2]
            self.grids[k] = grid
        grid = grid.view(1, -1, 2)  # 对所有的网格进行了堆叠 [1,6400,2]

        output = output.flatten(start_dim=2).permute(0, 2, 1)  # 对输出结果进行了堆叠[bs,6400,6]
        # 注意这里没有将偏移参数带入sigmoid函数。
        output[..., :2] = (output[..., :2] + grid.type_as(
            output)) * stride  # 利用预测参数对网格中心点进行了偏移，并且乘上了步距，也就是说这里得到的中心点是相对原图的，
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride  # 得到相对原图的wh
        return output, grid

    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels,
                   outputs, focal_loss=True):  # 这里的outputs将三个特征层的预测框都堆叠起来了 shape=[bs,all_box,6]
        # -----------------------------------------------#
        #   [batch, n_anchors_all, 4]
        # -----------------------------------------------#
        bbox_preds = outputs[:, :, :4]
        # -----------------------------------------------#
        #   [batch, n_anchors_all, 1]
        # -----------------------------------------------#
        obj_preds = outputs[:, :, 4:5]
        # -----------------------------------------------#
        #   [batch, n_anchors_all, n_cls]
        # -----------------------------------------------#
        cls_preds = outputs[:, :, 5:]

        total_num_anchors = outputs.shape[1]
        # -----------------------------------------------#
        #   x_shifts            [1, n_anchors_all]
        #   y_shifts            [1, n_anchors_all]
        #   expanded_strides    [1, n_anchors_all]
        # -----------------------------------------------#
        x_shifts = torch.cat(x_shifts, 1).type_as(outputs)  # 堆叠，每一个元素代表一个网格的x坐标
        y_shifts = torch.cat(y_shifts, 1).type_as(outputs)
        expanded_strides = torch.cat(expanded_strides, 1).type_as(outputs)  # 每一个元素代表一个网格的步距

        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        for batch_idx in range(outputs.shape[0]):  # 遍历当前batch
            num_gt = len(labels[batch_idx])  # GT框的数量
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                # -----------------------------------------------#
                #   gt_bboxes_per_image     [num_gt, 4]  当前图片的gt框坐标
                #   gt_classes              [num_gt,]  当前图片gt框的类别索引
                #   bboxes_preds_per_image  [8400, 4]  当前图片所有预测框的坐标
                #   cls_preds_per_image     [n_anchors_all, num_classes]  当前图片所有预测框的类别概率
                #   obj_preds_per_image     [n_anchors_all, 1]  当前图片所有预测框的confidence
                # -----------------------------------------------#
                gt_bboxes_per_image = labels[batch_idx][..., :4].type_as(outputs)  # gt框坐标
                gt_classes = labels[batch_idx][..., 4].type_as(outputs)  # gt框类别
                bboxes_preds_per_image = bbox_preds[batch_idx]  # '当前图片'所有的预测框坐标[8400,4]
                cls_preds_per_image = cls_preds[batch_idx]  # 类别概率
                obj_preds_per_image = obj_preds[batch_idx]  # 置信度
                # ----------
                #   gt_matched_classes [positive, ]  表示正样本对应的类别索引
                #   fg_mask[8400,] True or False 来表示这个预测框是否为正样本
                #   pred_ious_this_matching[num_positive,] 表示正样本与其gt框的iou大小
                #   matched_gt_inds[num_positive] 表示正样本对应gt框的索引
                #   num_fg_img=num_positve 表示正样本的个数
                # -----------
                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(
                    num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image,
                    cls_preds_per_image, obj_preds_per_image,
                    expanded_strides, x_shifts, y_shifts,
                )
                torch.cuda.empty_cache()
                num_fg += num_fg_img  # 用于计量当前batch所有的正样本框个数
                # ---------------------
                # cls_target[num_positive,num_classes]类别标签 首先根据每个正样本对应的类别生成one-hot编码，然后乘上正样本与gt框的iou大小。这里的真值索引处标签值不再是1 要注意
                # obj_target[8400,1] 所有预测框的置信度标签，正样本标签为1，负样本标签为0，所有的预测框参与置信度损失计算
                # reg_target[num_positive,4] 位置标签，首先根据正样本对应的gt框索引去找到对应的gt框生成位置标签。
                # ----------------------
                cls_target = F.one_hot(gt_matched_classes.to(torch.int64),
                                       self.num_classes).float() * pred_ious_this_matching.unsqueeze(
                    -1)  # [num_pre,num_class]
                obj_target = fg_mask.unsqueeze(-1)  # 置信度的标签[8400,1], 至此fg_mask只在细样本处标记为true
                reg_target = gt_bboxes_per_image[matched_gt_inds]  # [num_pre,4] 位置标签
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type(cls_target.type()))
            fg_masks.append(fg_mask)
        # 对当前batch所有的标签值进行堆叠，计算损失时统一计算
        cls_targets = torch.cat(cls_targets, 0)  # 将当前batch的框都堆叠
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        num_fg = max(num_fg, 1)  # 当前batch所有的正样本框数量
        # bbox_preds.view(-1, 4)[fg_mask] 先将当前batch所有预测框进行了堆叠，然后fg_masks筛选出了正样本 shape=[in_batch_all_细样本,4]
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum()  # 返回的是[num_正样本,]当前batch所有正样本的iou_loss
        # nn.BCEWithLogitsLoss会将预测值先经过一个sigmoid函数然后再计算bce_loss计算
        if focal_loss:
            obj_mask = fg_masks.unsqueeze(-1)  # [all_anchor,1]
            pos_neg_ratio = torch.where(obj_mask, torch.ones_like(obj_mask) * self.alpha,
                                        torch.ones_like(obj_mask) * (1 - self.alpha))
            conf = torch.sigmoid(obj_preds.view(-1, 1))  # 经过sigmoid后的置信度分数
            hard_easy_ratio = torch.where(obj_mask, torch.ones_like(conf) - conf,
                                          conf) ** self.gamma
            loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets) * pos_neg_ratio * hard_easy_ratio).sum() * self.focal_loss_ratio  # 2433
        else:
            loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum()  # 求所有预测框的置信度损失，采用bceloss
        # cls_targets[num_positive,num_classes]
        # cls_preds.view(-1, self.num_classes)[fg_masks] = [num_positive,num_classes] 逐一计算交叉熵损失
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum()
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls

        return loss / num_fg

    @torch.no_grad()
    def get_assignments(self, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image,
                        cls_preds_per_image, obj_preds_per_image, expanded_strides, x_shifts, y_shifts):
        # -------------------------------------------------------#
        #   fg_mask                 [n_anchors_all] bool 用来表示gird的中心是否落在某个gt内或center 初筛选正样本
        #   is_in_boxes_and_center  [num_gt, len(fg_mask)] 如果为true说明这个grid的中心即落在这个gt框内也落在这个gt的center内，注意，这里是为同一个gt
        #   也就说那些落在gt框1 也落在gt框2的 center 是为false的，而fg_mask是只要落在任意的gt框内或者gt框的center内都为true了
        # -------------------------------------------------------#
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts,
                                                                 y_shifts, total_num_anchors, num_gt)

        # -------------------------------------------------------#
        #   fg_mask                 [n_anchors_all]
        #   bboxes_preds_per_image  [fg_mask, 4]
        #   cls_preds_              [fg_mask, num_classes]
        #   obj_preds_              [fg_mask, 1]
        # -------------------------------------------------------#
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]  # 过滤掉那些不符合fg_mask的预测框
        cls_preds_ = cls_preds_per_image[fg_mask]
        obj_preds_ = obj_preds_per_image[fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]  # 符合fg_mask的预测框个数

        # -------------------------------------------------------#
        #   pair_wise_ious      [num_gt, fg_mask]
        # -------------------------------------------------------#
        # shape =[num_gt,num_pre],返回了每一个gt框和预测框 两两之间的iou
        pair_wise_ious = self.bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        # -------------------------------------------------------#
        #   cls_preds_          [num_gt, num_pre, num_classes]
        #   gt_cls_per_image    [num_gt, num_pre, num_classes]
        #   pair_wise_cls_loss  [num_gt, num_pre]
        # -------------------------------------------------------#
        if self.fp16:
            with torch.cuda.amp.autocast(enabled=False):
                cls_preds_ = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.unsqueeze(
                    0).repeat(num_gt, 1, 1).sigmoid_()  #
                # F.one_hot传入每个gt框的类别索引 和 总共的类别数，会给每一个gt框生成一个one_hot编码
                gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(
                    1, num_in_boxes_anchor, 1)
                pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(
                    -1)  # 每个gt框和每个预测框之间 两两的类别分数损失
        else:
            cls_preds_ = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.unsqueeze(
                0).repeat(num_gt, 1, 1).sigmoid_()
            gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1,
                                                                                                                   num_in_boxes_anchor,
                                                                                                                   1)
            pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
            del cls_preds_
        # cost矩阵 shape = [num_gt,num_pre] 体现了每个gt和每个预测框 之间的匹配程度
        #   pair_wise_ious_loss是每个gt框和粗正样本两两之间的iou loss,pair_wise_cls_loss是gt框和粗正样本两两之间的类别分数BCEloss
        #   ~is_in_boxes_and_center用于筛选掉那些没有同时落在gt框内及其center内的框。
        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center).float()

        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = self.dynamic_k_matching(cost,
                                                                                                       pair_wise_ious,
                                                                                                       gt_classes,
                                                                                                       num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

    # 返回box_a和box_b两两之间的iou
    def bboxes_iou(self, bboxes_a, bboxes_b, xyxy=True):
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            raise IndexError

        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            # box_a.shape=[num_gt,4] box_b.shape[num_pre,4]
            tl = torch.max(  # shape=[num_gt,num_pre,2]
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),  # 得到gt框的xmin,ymin
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),  # 得到预测框的xmin,ymin
            )  # gt框与预测框逐一相比较，得到gt框和预测框中xmin和ymin中较大的一个
            br = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )  # 得到gt框和预测框中xmax和ymax中较小的一个
            # torch.prod 返回输入张量给定维度上每行的积
            area_a = torch.prod(bboxes_a[:, 2:], 1)  # gt框的面积 num_gt,
            area_b = torch.prod(bboxes_b[:, 2:], 1)  # 预测框的面积   num_pre,
        en = (tl < br).type(tl.type()).prod(dim=2)  # num_pre,
        area_i = torch.prod(br - tl, 2) * en  # [num_gt,num_pre] 每个gt框与每个pre框的相交的面积
        return area_i / (area_a[:, None] + area_b - area_i)

    # 得到粗样本和 即在gt框内也在center内的anchor
    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt,
                          center_radius=2.5):
        # -------------------------------------------------------#
        #   expanded_strides_per_image  [n_anchors_all]
        #   x_centers_per_image         [num_gt, n_anchors_all]
        #   x_centers_per_image         [num_gt, n_anchors_all]
        # -------------------------------------------------------#
        expanded_strides_per_image = expanded_strides[0]  # 步距[8400,]
        # 将网格左上角点进行偏移0.5，然后乘上步距。就得到了相对原图的中心点。同时这里进行了repeat-> shape = [num_gt,8400]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)

        # -------------------------------------------------------#
        #   gt_bboxes_per_image_x       [num_gt, 8400]
        #   先是得到每个gt框的xmin,xmax,ymin,ymax，然后repeat->8400次
        # -------------------------------------------------------#
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)

        # -------------------------------------------------------#
        #   bbox_deltas     [num_gt, n_anchors_all, 4]
        # -------------------------------------------------------#
        b_l = x_centers_per_image - gt_bboxes_per_image_l  # 网格中心点x坐标 减去 gt框的xin坐标
        b_r = gt_bboxes_per_image_r - x_centers_per_image  # gt框的xmax坐标 减去 网格的中心点x坐标
        b_t = y_centers_per_image - gt_bboxes_per_image_t  # 网格中心点的y坐标 减去 gt框的ymin坐标
        b_b = gt_bboxes_per_image_b - y_centers_per_image  # gt框的ymax坐标 减去网格中心点的y坐标
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)  # shape = [num_gt,8400,4]

        # -------------------------------------------------------#
        #   is_in_boxes     [num_gt, 8400] 是一个bool类型，判断哪些中心点落在gt框内
        #   is_in_boxes_all [8400，]
        # -------------------------------------------------------#
        # 判断最后一维的4个数是否都大于0. 只有中心点落在这个gt框中间才满足
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0  # [num_gt,8400]
        # 该变量用于判断某个grid的中心点是否落在gt框内
        is_in_boxes_all = is_in_boxes.sum(
            dim=0) > 0  # 将两个gt框的结果相加，比如说有2个gt框。如果得到false，说明这个中心点没有落在一个gt框内，如果为true，说明落在了一个框内。

        # [num_gt,8400]  gt框的中心点x坐标 - 步距 * 2.5, 对于浅层的特征感受野小所以范围就小，深层的特征感受野大，所以范围就大。
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1,
                                                                                total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(
            0)
        # gt框的中心点x坐标 + 步距 * 2.5
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1,
                                                                                total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(
            0)
        # gt框的中心点y坐标 - 步距 * 2.5
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1,
                                                                                total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(
            0)
        # gt框的中心点y坐标 + 步距 * 2.5
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1,
                                                                                total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(
            0)

        # -------------------------------------------------------#
        #   center_deltas   [num_gt, n_anchors_all, 4]
        # -------------------------------------------------------#
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)

        # -------------------------------------------------------#
        #   is_in_centers       [num_gt, n_anchors_all]
        #   is_in_centers_all   [n_anchors_all]
        # -------------------------------------------------------#
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # 至此有两种判断正样本了：
        # 1.判断哪些grid的中心点落在gt框内。
        # 2.由gt框的中心点出发，对于80 * 80的grid cell,在gt框中心附近划了一个2.5 * 8的框。然后判断哪些grid的中心点落在这个框内。
        #   对于其它大小的特征层相同。

        # -------------------------------------------------------#
        #   is_in_boxes             [num_gt,8400]  bool  哪些点在GT框内
        #   is_in_centers           [num_gt,8400]  bool  哪些点在center内
        #   is_in_boxes_anchor      [8400]  # 哪些点在gt或center内
        #   is_in_boxes_and_center  [num_gt, is_in_boxes_anchor]
        # -------------------------------------------------------#
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all  # 取并集，也就是判断哪些grid的中心是满足上面两种判断方式之一的
        # is_in_boxes[:, is_in_boxes_anchor]可以筛选掉没落到gt框也没落到center的grid,这里剩下的是[num_gt,num_pre]也就是剩下至少在一个框内的
        is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        return is_in_boxes_anchor, is_in_boxes_and_center

    # 得到细样本
    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # -------------------------------------------------------#
        #   cost                [num_gt, fg_mask]
        #   pair_wise_ious      [num_gt, fg_mask]
        #   gt_classes          [num_gt]        
        #   fg_mask             [n_anchors_all]
        #   matching_matrix     [num_gt, fg_mask]
        # -------------------------------------------------------#
        matching_matrix = torch.zeros_like(cost)

        # ------------------------------------------------------------#
        #   选取iou最大的n_candidate_k个点
        #   然后求和，判断应该有多少点用于该框预测
        #   topk_ious           [num_gt, n_candidate_k]
        #   dynamic_ks          [num_gt]
        #   matching_matrix     [num_gt, fg_mask]
        # ------------------------------------------------------------#
        n_candidate_k = min(10, pair_wise_ious.size(1))
        # 返回预测框与gt框前10大的iou值
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        # 横向求和，得到为每个gt框分配正样本的个数
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        for gt_idx in range(num_gt):  # 遍历gt框
            # ------------------------------------------------------------#
            #   给每个真实框选取最小的动态k个点
            # ------------------------------------------------------------#
            #   为gt框选取k个cost最小的预测框，得到的预测框对应的anchor的索引
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[gt_idx][pos_idx] = 1.0  # 赋上标记
        del topk_ious, dynamic_ks, pos_idx

        # ------------------------------------------------------------#
        #   anchor_matching_gt  [fg_mask]
        # ------------------------------------------------------------#
        anchor_matching_gt = matching_matrix.sum(0)  # 纵向求和，判断是否出现一个anchor分配给多个gt框的情况
        if (anchor_matching_gt > 1).sum() > 0:
            # ------------------------------------------------------------#
            #   当某一个特征点指向多个真实框的时候
            #   选取cost最小的真实框。
            # ------------------------------------------------------------#
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        # ------------------------------------------------------------#
        #   fg_mask_inboxes  [fg_mask]
        #   num_fg为正样本的特征点个数
        # ------------------------------------------------------------#
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0  # 得到哪些anchor是作为精确样本的
        num_fg = fg_mask_inboxes.sum().item()

        # ------------------------------------------------------------#
        #   对fg_mask进行更新
        #   fg_mask[fg_mask.clone()]这一步是取出粗样本，注意在赋值时这样的表示 是表示在对于的位置进行赋值操作，不要去想着先把他取出来
        # ------------------------------------------------------------#
        fg_mask[fg_mask.clone()] = fg_mask_inboxes  # 此时f_mask是精筛选后的

        # ------------------------------------------------------------#
        #   获得特征点对应的物品种类
        #   matched_gt_inds [num_positive,] 表示每个正样本对应哪个gt框，这里存放的时gt框的顺序索引，如[3,1,0,2]表示第一个正样本对应第四个gt框
        #   gt_matched_classes[num_positive,]  表示每个正样本对应的类别索引
        # ------------------------------------------------------------#
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)  # 得到精筛选后的正样本 分别是对应哪个gt框的
        gt_matched_classes = gt_classes[matched_gt_inds]  # 根据gt框的类别，来得到正样本对应的类别， 正样本的类别应该与其对应的gt框的类别相同
        # matching_matrix * pair_wise_ious 只得到正样本处和gt框的Iou
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]  # 得到正样本与其gt框的iou大小
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA  深拷贝后的模型
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            ema_dic = self.ema.state_dict()
            for k, v in ema_dic.items():  # ema是原model深拷贝出来的，
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()  # v = v * d + (1-d) * v‘

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
