import colorsys
import os
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from tqdm import tqdm

from nets.yolo4 import YoloBody
from utils.utils import (DecodeBox, bbox_iou, letterbox_image,
                         non_max_suppression, yolo_correct_boxes)
from yolo import YOLO

'''
该FPS测试不包括前处理（归一化与resize部分）、绘图。
包括的内容为：网络推理、得分门限筛选、非极大抑制。
使用'img/street.jpg'图片进行测试，该测试方法参考库https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

video.py里面测试的FPS会低于该FPS，因为摄像头的读取频率有限，而且处理过程包含了前处理和绘图部分。
'''
class FPS_YOLO(YOLO):
    def get_FPS(self, image, test_interval):
        # 调整图片使其符合输入要求
        image_shape = np.array(np.shape(image)[0:2])

        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1],self.model_image_size[0])))
        else:
            crop_img = image.convert('RGB')
            crop_img = crop_img.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
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
            outputs = self.net(images)
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, len(self.class_names),
                                                    conf_thres=self.confidence,
                                                    nms_thres=self.iou)
            try:
                batch_detections = batch_detections[0].cpu().numpy()
                top_index = batch_detections[:,4]*batch_detections[:,5] > self.confidence
                top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
                top_label = np.array(batch_detections[top_index,-1],np.int32)
                top_bboxes = np.array(batch_detections[top_index,:4])
                top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)
                
                if self.letterbox_image:
                    boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)
                else:
                    top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                    top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                    top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                    top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                    boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)
            
            except:
                pass
                
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                outputs = self.net(images)
                output_list = []
                for i in range(3):
                    output_list.append(self.yolo_decodes[i](outputs[i]))
                output = torch.cat(output_list, 1)
                batch_detections = non_max_suppression(output, len(self.class_names),
                                                        conf_thres=self.confidence,
                                                        nms_thres=self.iou)
                try:
                    batch_detections = batch_detections[0].cpu().numpy()
                    top_index = batch_detections[:,4]*batch_detections[:,5] > self.confidence
                    top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
                    top_label = np.array(batch_detections[top_index,-1],np.int32)
                    top_bboxes = np.array(batch_detections[top_index,:4])
                    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)
                   
                    if self.letterbox_image:
                        boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)
                    else:
                        top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                        top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                        top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                        top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                        boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)
                
                except:
                    pass

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
        
yolo = FPS_YOLO()
test_interval = 100
img = Image.open('img/street.jpg')
tact_time = yolo.get_FPS(img, test_interval)
print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
