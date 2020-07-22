## 常见的问题汇总：

问：up主，可以给我发一份代码吗，代码在哪里下载啊？ 
答：Github上的地址就在视频简介里。复制一下就能进去下载了。

问：为什么我安装了tensorflow-gpu但是却没用利用GPU进行训练呢？
答：有没有用到GPU应该看终端里的提示，不要看任务管理器。

问：up主，为什么我下载的代码里面，model_data下面没有.pth或者.h5文件？ 
答：我一般会把权值上传到百度网盘，在GITHUB的README里面就能找到。

问：up主，为什么运行train.py会提示shape不匹配啊？
答：因为你训练的种类和原始的种类不同，网络结构会变化，所以最尾部的shape会有少量不匹配。

问：为什么提示说no module name utils.utils（no module name nets.yolo、no module name nets.ssd等一系列问题）啊？
答：根目录不对，查查相对目录的概念。查了基本上就明白了。

问：为什么提示说no module name matplotlib（no module name PIL）？
答：打开命令行安装就好。pip install matplotlib

问：为什么我运行predict.py会提示我说shape不匹配呀。（为什么我训练完了不能预测啊）
答：原因主要有仨：
1、在ssd、FasterRCNN里面，可能是train.py里面的num_classes没改。
2、model_path没改。
3、classes_path没改。
请检查清楚了！确定自己所用的model_path和classes_path是对应的！训练的时候用到的num_classes或者classes_path也需要检查！因为都是这个问题，如果来询问我，我也是同样的回答（问的太多了，但出错原因都是一样的，我在视频里也会强调需要修改model_path和classes_path，请一定注意修改。）。

问：为什么我运行train.py下面的命令行闪的贼快，还提示OOM啥的？ 
答：爆显存了，可以改小batch_size，如果batch_size=1才能运行的话，那么直接换网络吧，SSD的显存占用率是最小的，建议用SSD；
2G显存：SSD
4G显存：YOLOV3 Faster RCNN
6G显存：YOLOV4 Retinanet M2det Efficientdet等
8G+显存：随便选吧

问：为什么要冻结训练和解冻训练呀？
答：这是迁移学习的思想，因为神经网络主干特征提取部分所提取到的特征是通用的，我们冻结起来训练可以加快训练效率，也可以防止权值被破坏。

问：为什么我的LOSS一直不下降呀？
答：主要有五：
1、	数据集过少，小于500的自行考虑增加数据集。
2、	是否解冻训练。
3、	如果是yoloV4可以考虑关闭mosaic，mosaic不适用所有的情况。
4、	网络不适应，比如SSD不适合小目标，因为先验框固定了。
5、	不同网络的LOSS不同，LOSS只是一个参考指标，用于查看网络是否收敛，而非评价网络好坏，我的yolo代码都没有归一化，所以LOSS值看起来比较高，LOSS的值不重要，重要的是是否收敛！

问：能不能说说怎么绘制PR曲线啥的呀。
答：可以看mAP视频，结果里面有PR曲线。

问：我已经训练过几个世代了，能不能从这个基础上继续开始训练
答：可以，你在训练前，和载入预训练权重一样载入训练过的权重就行了。

问：能不能训练灰度图啊？
答：可以尝试一下在get_random_data里面将Image.open后的结果转换成RGB，预测的时候也这样试试。（仅供参考）

问：怎么用摄像头检测呀？
答：基本上所有目标检测库都有video.py可以进行摄像头检测，也有视频详细解释了摄像头检测的思路。

问：ubuntu下可以用吗？
答：可以。

问：怎么进行多GPU训练？
答：这个直接百度就好了，实现并不复杂。

问：为什么提示TypeError: cat() got an unexpected keyword argument 'axis'，Traceback (most recent call last)，AttributeError: 'Tensor' object has no attribute 'bool'？
答：这是版本问题，建议使用torch1.2以上版本

其它有很多稀奇古怪的问题，很多是版本问题，建议按照我的视频教程安装Keras和tensorflow。比如装的是tensorflow2，就不用问我说为什么我没法运行Keras-yolo啥的。那是必然不行的。


