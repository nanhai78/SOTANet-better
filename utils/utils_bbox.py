import numpy as np
import torch
from torchvision.ops import nms, boxes


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    # -----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    # -----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        # -----------------------------------------------------------------#
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        # -----------------------------------------------------------------#
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))  # 将wh等比例进行缩放，其中较长的边大小为640。
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape  # 宽高的放缩比

        box_yx = (box_yx - offset) * scale  # 图像再进入网络前，先进行了缩放再进行了偏移(偏移到灰图的中心) 那么返回到原图就要先进行偏移 再进行缩放
        box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)  # map到原图
    return boxes


def decode_outputs(outputs, input_shape):
    grids = []
    strides = []
    hw = [x.shape[-2:] for x in outputs]  # 特征层的宽高
    # ---------------------------------------------------#
    #   outputs输入前代表每个特征层的预测结果 是一个list,包含3个元素分别为三个特征层预测的结果
    #   batch_size, 4 + 1 + num_classes, 80, 80 => batch_size, 4 + 1 + num_classes, 6400
    #   batch_size, 5 + num_classes, 40, 40
    #   batch_size, 5 + num_classes, 20, 20
    #   batch_size, 4 + 1 + num_classes, 6400 + 1600 + 400 -> batch_size, 4 + 1 + num_classes, 8400
    #   堆叠后为batch_size, 8400, 5 + num_classes
    # ---------------------------------------------------#
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)  # 对三个特征层进行了堆叠，堆叠后shape=[bs,8400,85]
    # ---------------------------------------------------#
    #   获得每一个特征点属于每一个种类的概率
    # ---------------------------------------------------#
    outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])  # [bs,num_class,8400]
    for h, w in hw:
        # ---------------------------#
        #   根据特征层的高宽生成网格点
        # ---------------------------#
        grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])
        # ---------------------------#
        #   1, 6400, 2
        #   1, 1600, 2
        #   1, 400, 2
        # ---------------------------#
        grid = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)  # 先进行张量的拼接，在进行堆叠
        shape = grid.shape[:2]

        grids.append(grid)  # 存放三个特征层的网格点左上角坐标点
        # torch.full(size, value) 创建一个大小为size的矩阵，矩阵的元素全为value
        strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h))  # 存放三个特征层的步距
    # ---------------------------#
    #   将网格点堆叠到一起
    #   1, 6400, 2
    #   1, 1600, 2
    #   1, 400, 2
    #
    #   1, 8400, 2
    # ---------------------------#
    grids = torch.cat(grids, dim=1).type(outputs.type())
    strides = torch.cat(strides, dim=1).type(outputs.type())
    # ------------------------#
    #   根据网格点进行解码
    # ------------------------#
    outputs[..., :2] = (outputs[..., :2] + grids) * strides  # 得到预测框在原图的中心点xy
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides  # 得到预测框在原图的宽高wh
    # -----------------#
    #   归一化
    # -----------------#
    outputs[..., [0, 2]] = outputs[..., [0, 2]] / input_shape[1]
    outputs[..., [1, 3]] = outputs[..., [1, 3]] / input_shape[0]
    return outputs


def non_max_suppression(prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5,
                        nms_thres=0.4):
    # ----------------------------------------------------------#
    #   将预测结果的格式转换成左上角右下角的格式。
    #   prediction  [batch_size, 8400, 85]  且xywh进行了归一化
    # ----------------------------------------------------------#
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]  # 根据bs大小，生成长度为bs的list用于存放每张图片结果
    # ----------------------------------------------------------#
    #   对当前batch的预测结果进行循环
    # ----------------------------------------------------------#
    for i, image_pred in enumerate(prediction):
        # ----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    类别索引
        # ----------------------------------------------------------#
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        # ----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        # ----------------------------------------------------------#
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        if not image_pred.size(0):
            continue
        # -------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        # -------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]  # 第一轮筛选后剩下的框

        nms_out_index = boxes.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thres,
        )  # 得到经过nms后的box的索引，这里的索引是相对第一轮筛选后剩下的框的

        output[i] = detections[nms_out_index]

        # #------------------------------------------#
        # #   获得预测结果中包含的所有种类
        # #------------------------------------------#
        # unique_labels = detections[:, -1].cpu().unique()

        # if prediction.is_cuda:
        #     unique_labels = unique_labels.cuda()
        #     detections = detections.cuda()

        # for c in unique_labels:
        #     #------------------------------------------#
        #     #   获得某一类得分筛选后全部的预测结果
        #     #------------------------------------------#
        #     detections_class = detections[detections[:, -1] == c]

        #     #------------------------------------------#
        #     #   使用官方自带的非极大抑制会速度更快一些！
        #     #------------------------------------------#
        #     keep = nms(
        #         detections_class[:, :4],
        #         detections_class[:, 4] * detections_class[:, 5],
        #         nms_thres
        #     )
        #     max_detections = detections_class[keep]

        #     # # 按照存在物体的置信度排序
        #     # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
        #     # detections_class = detections_class[conf_sort_index]
        #     # # 进行非极大抑制
        #     # max_detections = []
        #     # while detections_class.size(0):
        #     #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
        #     #     max_detections.append(detections_class[0].unsqueeze(0))
        #     #     if len(detections_class) == 1:
        #     #         break
        #     #     ious = bbox_iou(max_detections[-1], detections_class[1:])
        #     #     detections_class = detections_class[1:][ious < nms_thres]
        #     # # 堆叠
        #     # max_detections = torch.cat(max_detections).data

        #     # Add max detections to outputs
        #     output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        if output[i] is not None:
            output[i] = output[i].cpu().numpy()
            # 将xim,ymin,xmax,ymax转换成xywh的形式
            box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
            # 将预测框 map 回原图 且转换测左上角右下角的形式
            output[i][:, :4] = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    return output
