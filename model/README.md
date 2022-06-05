# 模型文件

本目录为组建所有模型动态图的代码文件，包含多流模型和所有baseline。

- 多流模型
- [FCN](https://readpaper.com/paper/2952632681) ：最基本的编码——解码结构，特点为全卷积神经网络；
- [SegNet](https://readpaper.com/paper/2963881378) ：和FCN的结构极为类似，区别在于上采样使用最大上池化操作；
- [U-Net](https://readpaper.com/paper/1901129140) ：最基本和最经典的医学分割模型；
- [PspNet](https://readpaper.com/paper/2560023338) ：以VGG或ResNet作为骨干网络，加入多层次卷积结构以捕捉不同尺寸的特征；
- [DeepLab](https://arxiv.org/abs/1706.05587) ： 以VGG或ResNet作为骨干网络，使用空洞卷积，在保持图像分辨率的同时能够扩大像素感受野；
- [bisenetv2](https://readpaper.com/paper/3014795891) ：轻量化语义分割网络

除此以外还包括损失函数的代码文件，内置针对不同分割头的损失函数。
