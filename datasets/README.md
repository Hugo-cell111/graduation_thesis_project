# 数据集介绍
首先在该目录下创建三个空目录：train, val和test，同时在三个空目录分别创建子目录img和gt，前者存储原始图像，后者存储分割后得到的二值化图像。

然后根据以下链接下载数据集，并自行放在该目录下。每个文件夹均同时包含img和gt的子目录，以保证代码正常执行。

## 数据集
- CrackTree260&CrackLS315&CRKWH100
百度网盘链接：https://pan.baidu.com/s/1PWiBzoJlc8qC8ffZu2Vb8w
提取码：zfoo
- Crack500&CFD
百度网盘链接：https://pan.baidu.com/share/init?surl=JwJO96BOtJ50MykBcYKknQ
提取码：jviq
- CrackForest
数据集和标签：https://github.com/cuilimeng/CrackForest-dataset

# 引用
- CrackTree260&CrackLS315&CRKWH100数据集
```
@article{zou2018deepcrack,
  title={Deepcrack: Learning Hierarchical Convolutional Features for Crack Detection},
  author={Zou, Qin and Zhang, Zheng and Li, Qingquan and Qi, Xianbiao and Wang, Qian and Wang, Song},
  journal={IEEE Transactions on Image Processing},
  volume={28},
  number={3},
  pages={1498--1512},
  year={2019},
}
```
- CrackForest&CFD
```
@article{cfd,
  title={Automatic Road Crack Detection Using Random Structured Forests},
  author={Yong Shi and Limeng Cui and Zhiquan Qi and Fan Meng and Zhensong Chen},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2016}
}
```
- Crack500
```
@article{fphbn,
  title={Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection},
  author={Fan Yang and Lei Zhang and Sijia Yu and Danil V. Prokhorov and Xue Mei and Haibin Ling},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2019}
}
```
