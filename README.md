## YOLOV4：You Only Look Once目标检测模型在pytorch当中的实现
---

### 目录
1. [所需环境 Environment](#所需环境)
2. [注意事项 Attention](#注意事项)
3. [小技巧的设置 TricksSet](#小技巧的设置)
4. [文件下载 Download](#文件下载))
5. [训练步骤 How2train](#训练步骤)
6. [参考资料 Reference](#Reference)

### YOLOV4的改进
- [x] 主干特征提取网络：DarkNet53 => CSPDarkNet53
- [x] 特征金字塔：SPP，PAN
- [x] 训练用到的小技巧：Mosaic数据增强、Label Smoothing平滑、CIOU、学习率余弦退火衰减
- [x] 激活函数：使用Mish激活函数
- [ ] ……balabla

### 所需环境
torch==1.2.0

### 注意事项
代码中的yolo4_weights.pth是基于608x608的图片训练的，但是由于显存原因。我将代码中的图片大小修改成了416x416。有需要的可以修改回来。 代码中的默认anchors是基于608x608的图片的。   

### 小技巧的设置
在train.py文件下：   
1、mosaic参数可用于控制是否实现Mosaic数据增强。   
2、Cosine_scheduler可用于控制是否使用学习率余弦退火衰减。   
3、label_smoothing可用于控制是否Label Smoothing平滑。

### 文件下载
训练所需的yolo4_weights.pth可在百度网盘中下载。  
链接: https://pan.baidu.com/s/1VNSYi39AaqjHVbdNpW_7sw 提取码: q2iv  
yolo4_weights.pth是coco数据集的权重。  
yolo4_voc_weights.pth是voc数据集的权重。

### 训练步骤
1、本文使用VOC格式进行训练。  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4、在训练前利用voc2yolo4.py文件生成对应的txt。  
5、再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。  
```python
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```
6、就会生成对应的2007_train.txt，每一行对应其图片位置及其真实框的位置。  
7、在训练前需要修改model_data里面的voc_classes.txt文件，需要将classes改成你自己的classes。  
8、运行train.py即可开始训练。

### mAP目标检测精度计算更新
更新了get_gt_txt.py、get_dr_txt.py和get_map.py文件。  
get_map文件克隆自https://github.com/Cartucho/mAP  
具体mAP计算过程可参考：https://www.bilibili.com/video/BV1zE411u7Vw

### Reference
https://github.com/qqwweee/keras-yolo3/  
https://github.com/Cartucho/mAP  
https://github.com/Ma-Dan/keras-yolo4  
