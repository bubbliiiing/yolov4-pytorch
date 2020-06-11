#-------------------------------------#
#       创建YOLO类
#-------------------------------------#
import cv2
import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
from nets.yolo4 import YoloBody
import torch.backends.cudnn as cudnn
from PIL import Image,ImageFont, ImageDraw
from torch.autograd import Variable
from utils.utils import non_max_suppression, bbox_iou, DecodeBox,letterbox_image,yolo_correct_boxes

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#--------------------------------------------#
class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo4_weights.pth',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "model_image_size" : (416, 416, 3),
        "confidence": 0.5,
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    #---------------------------------------------------#
    #   获得所有的先验框
    #---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1,:,:]

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        
        self.net = YoloBody(len(self.anchors[0]),len(self.class_names)).eval()

        # 加快模型训练的效率
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        
        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
    
        print('Finished!')

        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(DecodeBox(self.anchors[i], len(self.class_names),  (self.model_image_size[1], self.model_image_size[0])))


        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(image, (self.model_image_size[0],self.model_image_size[1])))
        photo = np.array(crop_img,dtype = np.float32)
        photo /= 255.0
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)
        images = np.asarray(images)

        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            
        output_list = []
        for i in range(3):
            output_list.append(self.yolo_decodes[i](outputs[i]))
        output = torch.cat(output_list, 1)
        batch_detections = non_max_suppression(output, len(self.class_names),
                                                conf_thres=self.confidence,
                                                nms_thres=0.3)
        try:
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return image
            
        top_index = batch_detections[:,4]*batch_detections[:,5] > self.confidence
        top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
        top_label = np.array(batch_detections[top_index,-1],np.int32)
        top_bboxes = np.array(batch_detections[top_index,:4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

        # 去掉灰条
        boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0]

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

