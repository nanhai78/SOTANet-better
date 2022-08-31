import os
import random
import xml.etree.ElementTree as ET

import numpy as np

from utils.utils import get_classes

# --------------------------------------------------------------------------------------------------------------------------------#
#   annotation_mode用于指定该文件运行时计算的内容
#   annotation_mode为0代表整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面的txt以及训练用的2007_train.txt、2007_val.txt
#   annotation_mode为1代表获得VOCdevkit/VOC2007/ImageSets里面的txt
#   annotation_mode为2代表获得训练用的2007_train.txt、2007_val.txt
# --------------------------------------------------------------------------------------------------------------------------------#
annotation_mode = 2
# -------------------------------------------------------------------#
#   必须要修改，用于生成2007_train.txt、2007_val.txt的目标信息
#   与训练和预测所用的classes_path一致即可
#   如果生成的2007_train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
#   仅在annotation_mode为0和2的时候有效
# -------------------------------------------------------------------#
classes_path = 'model_data/myclass.txt'
# --------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
#   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1
#   仅在annotation_mode为0和1的时候有效
# --------------------------------------------------------------------------------------------------------------------------------#
trainval_percent = 0.9
train_percent = 0.9
# -------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
# -------------------------------------------------------#
VOCdevkit_path = 'data'
xml_path = 'label22B'
bmp_path = 'bmp11B'
VOCdevkit_sets = ['train', 'val']
classes, _ = get_classes(classes_path)

# -------------------------------------------------------#
#   统计目标数量
# -------------------------------------------------------#
photo_nums = np.zeros(len(VOCdevkit_sets))
nums = np.zeros(len(classes))


def convert_annotation(image_id, list_file):   # image_id文件名
    # 打开xml文件
    in_file = open(os.path.join(VOCdevkit_path, 'Annotations/%s.xml' % image_id), encoding='utf-8')
    # 读取xml文件
    tree = ET.parse(in_file)
    # 获取根节点
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text  # 获取obj的name
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)  # 得到类别的索引
        xmlbox = obj.find('bndbox')
        # 得到边界框坐标
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))  # 写入坐标 加类别索引

        nums[classes.index(cls)] = nums[classes.index(cls)] + 1  # 统计每个类别的目标数量


if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")
    # 根据annotations里面的xml文件，来生成test,train,trainval,val.txt文件
    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        xmlfilepath = os.path.join(VOCdevkit_path, xml_path)
        saveBasePath = os.path.join(VOCdevkit_path, 'Main')
        temp_xml = os.listdir(xmlfilepath)  # 得到该目录下的文件名
        total_xml = []  # 存放xml文件名的列表
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num = len(total_xml)  # xml文件数量
        list = range(num)
        tv = int(num * trainval_percent)  # 获取训练验证集的数量 2025
        tr = int(tv * train_percent)  # 获取训练集的数量 1822
        trainval = list[:tv]
        # trainval = random.sample(list, tv)  # 从list中随机选取tv个元素，作为新列表，这里存放的是被选为训练验证样本的索引
        train = random.sample(trainval, tr)

        print("train and val size", tv)
        print("train size", tr)
        # 在imagesets/main下创建这4个文件夹
        ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'a')
        ftest = open(os.path.join(saveBasePath, 'test.txt'), 'a')
        ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'a')
        fval = open(os.path.join(saveBasePath, 'val.txt'), 'a')

        # 往各个txt文件写入xml的文件名
        for i in list:
            name = total_xml[i][:-4] + '\n'  # 遍历xml文件名
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()
        print("Generate txt in ImageSets done.")

    # 生成train.txt和val.txt文件，里面主要存了图片的绝对路径和 box的坐标信息和类别信息
    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate 2007_train.txt and 2007_val.txt for train.")
        type_index = 0
        for image_set in VOCdevkit_sets:  # VOCdevkit_sets = ['train', 'val']
            # 读取Main下的train.txt
            image_ids = open(os.path.join(VOCdevkit_path, 'Main/%s.txt' % image_set),
                             encoding='utf-8').read().strip().split()
            # list_file代表2007_train.txt文件
            list_file = open('%sSet.txt' % image_set, 'w', encoding='utf-8')
            for image_id in image_ids:
                # 在2007_train.txt文件中写入相应的图片绝对路径和名称，
                list_file.write('%s/BMPImage/%s.bmp' % (os.path.abspath(VOCdevkit_path), image_id))

                convert_annotation(image_id, list_file)  # 写入box坐标和类别
                list_file.write('\n')
            photo_nums[type_index] = len(image_ids)  # photo_nums[0]表示训练集的图像，photo_nums[1]表示验证集的图像
            type_index += 1
            list_file.close()
        print("Generate 2007_train.txt and 2007_val.txt for train done.")


        def printTable(List1, List2):
            for i in range(len(List1[0])):
                print("|", end=' ')
                for j in range(len(List1)):
                    print(List1[j][i].rjust(int(List2[j])), end=' ')
                    print("|", end=' ')
                print()


        str_nums = [str(int(x)) for x in nums]
        tableData = [
            classes, str_nums
        ]
        colWidths = [0] * len(tableData)  # clowidths记录了[类别名称长度，类别个数长度]
        len1 = 0
        for i in range(len(tableData)):  # 2
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    colWidths[i] = len(tableData[i][j])
        printTable(tableData, colWidths)

        if photo_nums[0] <= 500:
            print("训练集数量小于500，属于较小的数据量，请注意设置较大的训练世代（Epoch）以满足足够的梯度下降次数（Step）。")

        if np.sum(nums) == 0:
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("（重要的事情说三遍）。")
