'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
import cv2
from PIL import Image
import numpy as np

from yolo import YOLO

from IPython import embed

yolo = YOLO()

image = Image.open('./img/view2.jpg')# 返回PIL.img对象
uncroped_image = cv2.imread("./img/view2.jpg")

r_image,boxes = yolo.detect_image(image)

# 进行裁剪
box = boxes

for i in range(boxes.shape[0]):
    # top, left, bottom, right = boxes[i]
    # 或者用下面这句等价
    top = boxes[i][0]
    left = boxes[i][1]
    bottom = boxes[i][2]
    right = boxes[i][3]

    top = top - 5
    left = left - 5
    bottom = bottom + 5
    right = right + 5

    # 左上角点的坐标
    top = int(max(0, np.floor(top + 0.5).astype('int32')))

    left = int(max(0, np.floor(left + 0.5).astype('int32')))
    # 右下角点的坐标
    bottom = int(min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32')))
    right = int(min(np.shape(image)[1], np.floor(right + 0.5).astype('int32')))

    # embed()

    # 问题出在这里：不能用这个方法，看两个参数是长和宽，是从图像的原点开始裁剪的，这样肯定是不对的
    croped_region = uncroped_image[top:bottom,left:right]# 先高后宽
    # 将裁剪好的目标保存到本地
    cv2.imwrite("./output/croped_view2_img_"+str(i)+".jpg",croped_region)

# embed()
r_image.show()
