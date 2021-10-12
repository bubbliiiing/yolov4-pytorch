## YOLOV4：You Only Look Once目标检测模型在pytorch当中的实现
---

**2021年10月12日更新：**   
**进行了大幅度的更新，对代码的模块进行修改，加了大量注释。**   

**2021年2月7日更新：**   
**加入letterbox_image的选项，关闭letterbox_image后网络的map得到大幅度提升。**

## 目录
1. [性能情况 Performance](#性能情况)
2. [实现的内容 Achievement](#实现的内容)
3. [所需环境 Environment](#所需环境)
4. [注意事项 Attention](#注意事项)
5. [小技巧的设置 TricksSet](#小技巧的设置)
6. [文件下载 Download](#文件下载)
7. [训练步骤 How2train](#训练步骤)
8. [预测步骤 How2predict](#预测步骤)
9. [评估步骤 How2eval](#评估步骤)
10. [参考资料 Reference](#Reference)

## 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| VOC07+12+COCO | [yolo4_voc_weights.pth](https://github.com/bubbliiiing/yolov4-pytorch/releases/download/v1.0/yolo4_voc_weights.pth) | VOC-Test07 | 416x416 | - | 89.0 
| COCO-Train2017 | [yolo4_weights.pth](https://github.com/bubbliiiing/yolov4-pytorch/releases/download/v1.0/yolo4_weights.pth) | COCO-Val2017 | 416x416 | 46.1 | 70.2 


## 实现的内容
- [x] 主干特征提取网络：DarkNet53 => CSPDarkNet53
- [x] 特征金字塔：SPP，PAN
- [x] 训练用到的小技巧：Mosaic数据增强、Label Smoothing平滑、CIOU、学习率余弦退火衰减
- [x] 激活函数：使用Mish激活函数
- [ ] ……balabla

## 所需环境
torch==1.2.0

## 注意事项
代码中的yolo4_weights.pth是基于608x608的图片训练的，但是由于显存原因。我将代码中的图片大小修改成了416x416。有需要的可以修改回来。 代码中的默认anchors是基于608x608的图片的。   
**注意不要使用中文标签，文件夹中不要有空格！**   
**在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类，在train.py中将classes_path指向该文件**。  

## 小技巧的设置
在train.py文件下：   
1、mosaic参数可用于控制是否实现Mosaic数据增强。   
2、Cosine_scheduler可用于控制是否使用学习率余弦退火衰减。   
3、label_smoothing可用于控制是否Label Smoothing平滑。

## 文件下载
训练所需的yolo4_weights.pth可在百度网盘中下载。  
链接: https://pan.baidu.com/s/1WlDNPtGO1pwQbqwKx1gRZA 提取码: p4sc  
yolo4_weights.pth是coco数据集的权重。  
yolo4_voc_weights.pth是voc数据集的权重。

VOC数据集下载地址如下，里面已经包括了训练集、测试集、验证集（与测试集一样），无需再次划分：  
链接: https://pan.baidu.com/s/1YuBbBKxm2FGgTU5OfaeC5A    
提取码: uack   

## 训练步骤
### a、训练VOC07+12数据集
1. 数据集的准备   
**本文使用VOC格式进行训练，训练前需要下载好VOC07+12的数据集，解压后放在根目录**  

2. 数据集的处理   
修改voc_annotation.py里面的annotation_mode=2，运行voc_annotation.py生成根目录下的2007_train.txt和2007_val.txt。   

3. 开始网络训练   
train.py的默认参数用于训练VOC数据集，直接运行train.py即可开始训练。   

4. 训练结果预测   
训练结果预测需要用到两个文件，分别是yolo.py和predict.py。我们首先需要去yolo.py里面修改model_path以及classes_path，这两个参数必须要修改。   
**model_path指向训练好的权值文件，在logs文件夹里。   
classes_path指向检测类别所对应的txt。**   
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。   

### b、训练自己的数据集
1. 数据集的准备  
**本文使用VOC格式进行训练，训练前需要自己制作好数据集，**    
训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。   
训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。   

2. 数据集的处理  
在完成数据集的摆放之后，我们需要利用voc_annotation.py获得训练用的2007_train.txt和2007_val.txt。   
修改voc_annotation.py里面的参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt。   
训练自己的数据集时，可以自己建立一个cls_classes.txt，里面写自己所需要区分的类别。   
model_data/cls_classes.txt文件内容为：      
```python
cat
dog
...
```
修改voc_annotation.py中的classes_path，使其对应cls_classes.txt，并运行voc_annotation.py。  

3. 开始网络训练  
**训练的参数较多，均在train.py中，大家可以在下载库后仔细看注释，其中最重要的部分依然是train.py里的classes_path。**  
**classes_path用于指向检测类别所对应的txt，这个txt和voc_annotation.py里面的txt一样！训练自己的数据集必须要修改！**  
修改完classes_path后就可以运行train.py开始训练了，在训练多个epoch后，权值会生成在logs文件夹中。  

4. 训练结果预测  
训练结果预测需要用到两个文件，分别是yolo.py和predict.py。在yolo.py里面修改model_path以及classes_path。  
**model_path指向训练好的权值文件，在logs文件夹里。  
classes_path指向检测类别所对应的txt。**  
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。  

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载yolo_weights.pth，放入model_data，运行predict.py，输入  
```python
img/street.jpg
```
2. 在predict.py里面进行设置可以进行fps测试和video视频检测。  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"        : 'model_data/yolo_weights.pth',
    "classes_path"      : 'model_data/coco_classes.txt',
    #---------------------------------------------------------------------#
    #   anchors_path代表先验框对应的txt文件，一般不修改。
    #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
    #---------------------------------------------------------------------#
    "anchors_path"      : 'model_data/yolo_anchors.txt',
    "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    #---------------------------------------------------------------------#
    #   输入图片的大小，必须为32的倍数。
    #---------------------------------------------------------------------#
    "input_shape"       : [416, 416],
    #---------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    #---------------------------------------------------------------------#
    "confidence"        : 0.5,
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    #---------------------------------------------------------------------#
    #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
    #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
    #---------------------------------------------------------------------#
    "letterbox_image"   : False,
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"              : True,
}
```
3. 运行predict.py，输入  
```python
img/street.jpg
```
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。  

## 评估步骤 
### a、评估VOC07+12的测试集
1. 本文使用VOC格式进行评估。VOC07+12已经划分好了测试集，无需利用voc_annotation.py生成ImageSets文件夹下的txt。
2. 在yolo.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**  
3. 运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

### b、评估自己的数据集
1. 本文使用VOC格式进行评估。  
2. 如果在训练前已经运行过voc_annotation.py文件，代码会自动将数据集划分成训练集、验证集和测试集。如果想要修改测试集的比例，可以修改voc_annotation.py文件下的trainval_percent。trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1。train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1。
3. 利用voc_annotation.py划分测试集后，前往get_map.py文件修改classes_path，classes_path用于指向检测类别所对应的txt，这个txt和训练时的txt一样。评估自己的数据集必须要修改。
4. 在yolo.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**  
5. 运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

## Reference
https://github.com/qqwweee/keras-yolo3/  
https://github.com/Cartucho/mAP  
https://github.com/Ma-Dan/keras-yolo4  
