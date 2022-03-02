#-------------------------------------------------------#
#   用于处理COCO数据集，根据json文件生成txt文件用于训练
#-------------------------------------------------------#
import json
import os
from collections import defaultdict

#-------------------------------------------------------#
#   指向了COCO训练集与验证集图片的路径
#-------------------------------------------------------#
train_datasets_path     = "coco_dataset/train2017"
val_datasets_path       = "coco_dataset/val2017"

#-------------------------------------------------------#
#   指向了COCO训练集与验证集标签的路径
#-------------------------------------------------------#
train_annotation_path   = "coco_dataset/annotations/instances_train2017.json"
val_annotation_path     = "coco_dataset/annotations/instances_val2017.json"

#-------------------------------------------------------#
#   生成的txt文件路径
#-------------------------------------------------------#
train_output_path       = "coco_train.txt"
val_output_path         = "coco_val.txt"

if __name__ == "__main__":
    name_box_id = defaultdict(list)
    id_name     = dict()
    f           = open(train_annotation_path, encoding='utf-8')
    data        = json.load(f)

    annotations = data['annotations']
    for ant in annotations:
        id = ant['image_id']
        name = os.path.join(train_datasets_path, '%012d.jpg' % id)
        cat = ant['category_id']
        if cat >= 1 and cat <= 11:
            cat = cat - 1
        elif cat >= 13 and cat <= 25:
            cat = cat - 2
        elif cat >= 27 and cat <= 28:
            cat = cat - 3
        elif cat >= 31 and cat <= 44:
            cat = cat - 5
        elif cat >= 46 and cat <= 65:
            cat = cat - 6
        elif cat == 67:
            cat = cat - 7
        elif cat == 70:
            cat = cat - 9
        elif cat >= 72 and cat <= 82:
            cat = cat - 10
        elif cat >= 84 and cat <= 90:
            cat = cat - 11
        name_box_id[name].append([ant['bbox'], cat])

    f = open(train_output_path, 'w')
    for key in name_box_id.keys():
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
    f.close()

    name_box_id = defaultdict(list)
    id_name     = dict()
    f           = open(val_annotation_path, encoding='utf-8')
    data        = json.load(f)

    annotations = data['annotations']
    for ant in annotations:
        id = ant['image_id']
        name = os.path.join(val_datasets_path, '%012d.jpg' % id)
        cat = ant['category_id']
        if cat >= 1 and cat <= 11:
            cat = cat - 1
        elif cat >= 13 and cat <= 25:
            cat = cat - 2
        elif cat >= 27 and cat <= 28:
            cat = cat - 3
        elif cat >= 31 and cat <= 44:
            cat = cat - 5
        elif cat >= 46 and cat <= 65:
            cat = cat - 6
        elif cat == 67:
            cat = cat - 7
        elif cat == 70:
            cat = cat - 9
        elif cat >= 72 and cat <= 82:
            cat = cat - 10
        elif cat >= 84 and cat <= 90:
            cat = cat - 11
        name_box_id[name].append([ant['bbox'], cat])

    f = open(val_output_path, 'w')
    for key in name_box_id.keys():
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
    f.close()
