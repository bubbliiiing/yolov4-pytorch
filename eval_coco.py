import colorsys
import json
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from tqdm import tqdm

from nets.yolo4 import YoloBody
from utils.utils import (DecodeBox, bbox_iou, diou_non_max_suppression,
                         letterbox_image, non_max_suppression,
                         yolo_correct_boxes)
from yolo import YOLO

coco_classes = {'person': 1, 'bicycle': 2, 'car': 3, 'motorbike': 4, 'aeroplane': 5, 
    'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10, 'fire hydrant': 11, 
    '': 83, 'stop sign': 13, 'parking meter': 14, 'bench': 15, 'bird': 16, 'cat': 17, 
    'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24, 
    'giraffe': 25, 'backpack': 27, 'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 
    'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39, 
    'baseball glove': 40, 'skateboard': 41, 'surfboard': 42, 'tennis racket': 43, 'bottle': 44, 
    'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 
    'apple': 53, 'sandwich': 54, 'orange': 55, 'broccoli': 56, 'carrot': 57, 'hot dog': 58, 
    'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62, 'sofa': 63, 'pottedplant': 64, 'bed': 65, 
    'diningtable': 67, 'toilet': 70, 'tvmonitor': 72, 'laptop': 73, 'mouse': 74, 'remote': 75, 
    'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81, 
    'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86, 'scissors': 87, 'teddy bear': 88, 
    'hair drier': 89, 'toothbrush': 90
}

class mAP_YOLO(YOLO):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image_id, image, results):
        self.confidence = 0.001
        image_shape = np.array(np.shape(image)[0:2])

        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        crop_img = np.array(letterbox_image(image, (self.model_image_size[1],self.model_image_size[0])))
        photo = np.array(crop_img,dtype = np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))
            if self.cuda:
                images = images.cuda()

            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))

            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, len(self.class_names),
                                                    conf_thres=self.confidence,
                                                    nms_thres=self.iou)

            #---------------------------------------------------------#
            #   如果没有检测出物体，返回原图
            #---------------------------------------------------------#
            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return image
            
            #---------------------------------------------------------#
            #   对预测框进行得分筛选
            #---------------------------------------------------------#
            top_index = batch_detections[:,4] * batch_detections[:,5] > self.confidence
            top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
            top_label = np.array(batch_detections[top_index,-1],np.int32)
            top_bboxes = np.array(batch_detections[top_index,:4])
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

            #-----------------------------------------------------------------#
            #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
            #   因此生成的top_bboxes是相对于有灰条的图像的
            #   我们需要对其进行修改，去除灰条的部分。
            #-----------------------------------------------------------------#
            boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)

        for i, c in enumerate(top_label):
            result = {}
            predicted_class = self.class_names[c]
            top, left, bottom, right = boxes[i]

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            result["image_id"] = int(image_id)
            result["category_id"] = coco_classes[predicted_class]
            result["bbox"] = [float(left),float(top),float(right-left),float(bottom-top)]
            result["score"] = float(top_conf[i])
            results.append(result)

        return results

yolo = mAP_YOLO()

jpg_names = os.listdir("./coco_dataset/val2017")

with open("./coco_dataset/eval_results.json","w") as f:
    results = []
    for jpg_name in tqdm(jpg_names):
        if jpg_name.endswith("jpg"):
            image_path = "./coco_dataset/val2017/" + jpg_name
            image = Image.open(image_path)
            # 开启后在之后计算mAP可以可视化
            results = yolo.detect_image(jpg_name.split(".")[0],image,results)
    json.dump(results,f)
