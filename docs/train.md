# 训练情况

## deeplab

### 训练过程

0. 刚开始时，发现Focalloss报错，经Debug后发现是因为传入focalloss的y_pred中存在部分数x < e (e -> 0);这些数在取log后导致出现nan，因此在模型尾部加入softmax，并在focalloss中log之前加1e-7；

1. 针对原本的代码进行训练，使用tr2_cropped训练集、epoch=50、不同loss进行训练，然后发现结果很差，如下图

<img src="../../dataset/dataset/tr2_cropped/data/1.png" width="50%" height="50%"><img src="../train/manual_check/img/deeplab/tr2/predict_img_lovasz.png" width="50%" height="50%">

2. 然后发现label是有0、1、2、3四种，所以调整代码，在训练之前对label独热化，使得label数据宽度为(1024, 1024, 4)，然后依然使用tr2_cropped训练集、epoch=50、不同loss进行训练，然后发现结果更差，如下图

<img src="../../dataset/dataset/tr2_cropped/data/1.png" width="50%" height="50%"><img src="../train/manual_check/img/deeplab/tr2/predict_img_lovasz_4.png" width="50%" height="50%">

3. 因为当前基于MobilenetV2的Deeplab模型参数量较大(大约8e7)，而tr2_cropped训练集很小，所以决定采用tr3_cropped训练集、epoch=50、不同loss进行训练，然后发现结果稍有好转，如下图

<img src="../../dataset/dataset/tr3_cropped/data/1.png" width="50%" height="50%"><img src="../train/manual_check/img/deeplab/tr3/lovasz/img1_ep50.png" width="50%" height="50%">

### 训练发现
0. 这个label太差了，比如这个原图和label的对比，植被的标注很奇怪；

<img src="../../dataset/dataset/tr3_cropped/data/1.png" width="50%" height="50%"><img src="../train/manual_check/img/label/label_img_1.png" width="50%" height="50%">

还有这个，比例差距太大；

<img src="../../dataset/dataset/tr3_cropped/data/3.png" width="50%" height="50%"><img src="../train/manual_check/img/label/label_img_3.png" width="50%" height="50%">

1. 从metrics上来说，交叉熵和Focalloss效果奇差，metrics基本不变甚至下降；Diceloss和Boundaryloss效果很差，metrics稍有变化；Lovaszloss效果很好，metrics平稳上升，因此一个训练方向就是继续加大数据集和epoch，看lovaszloss的metrics能否继续提高；

经检验lovasz的metrics确实能提高，但推理效果更差了，用训练集推理应该不会过拟合啊，很烦)，下图分别是ep=20、50、100的推理情况

<img src="../train/manual_check/img/deeplab/tr3/lovasz/img1_ep20.png" width="30%" height="30%"><img src="../train/manual_check/img/deeplab/tr3/lovasz/img1_ep50.png" width="30%" height="30%"><img src="../train/manual_check/img/deeplab/tr3/lovasz/img1_ep100.png" width="30%" height="30%">


2. 从手动推理效果来看，除了focalloss之外，其他的都有些像那么回事，因此可以尝试组合loss，另外focalloss大概率是写的有问题，需要改一下

3. 可以尝试使用Mobilenet的预训练权重，但是这里是直接resize图片的大小为(224, 224, 3)吗？


## Unet
一开始Unet的结构没有加上BatchNormalization层，导致训练的时候没有训了大约2个epochs之后loss不再下降，而作为metrics的mean_io_u也不再上升，这一点很奇怪，但是没有管，仍然硬train一发，这里batchsize = 1，可能会有使得模型的训练中存在噪声影响以及速度过慢等问题，model.compile采用:

```
optimizer = tf.keras.optimizers.Adam(),
loss = BinaryCrossentropy()，
metrics=tf.keras.metrics.MeanIoU(num_classes=4)
Unet模型config为:
{"model_config": {"min_kernel_num": 64, "num_classes": 4, "depth": 4}, "train_config": {"epochs": 20, "batch_size": null}}
```

如上，训练20epochs，手动推理查看效果，效果极差，这里给出对比图，图1是原图，图2是训练后的语义分割表现：

<img src="../../dataset/dataset/tr3_cropped/data/1.png" width="50%" height="50%"><img src="../train/manual_check/img/unet/predict_img1_BiEntropy_ep20.png" width="50%" height="50%">


那么这很明显模型没有学习到东西，结合之前loss不再下降的表现，初步猜想是梯度没有回传过去，导致只有最后的几层被更新，其余的前面的层没有接收到gradient。于是在原有的Unet模型中，在encoder和decoder的layer中都加上BatchNormalization层，又测试了一遍，发现问题似乎得到了解决。这里我们仍然采用同样的loss function，model.compile采用:

```
optimizer = tf.keras.optimizers.Adam(),
loss = BinaryCrossentropy(),
metrics=tf.keras.metrics.MeanIoU(num_classes=4)
Unet模型config为:
{"model_config": {"min_kernel_num": 64, "num_classes": 4, "depth": 4}, "train_config": {"epochs": 20, "batch_size": null}}
```

进行手动推理，有原图和语义分割结果比较如下：

<img src="../../dataset/dataset/tr3_cropped/data/1.png" width="50%" height="50%"><img src="../train/manual_check/img/unet/predict_img1_BiEntropyLoss_ep20.png" width="50%" height="50%">

发现效果很好，但BinaryCrossentropy()其实并不算特别出色的loss function，所以又换成了LovaszLoss()，仅仅train掉10个epochs，表现已经超出BinaryCrossentropy()作为loss function时的表现，其语义分割结果如下：

<img src="../../dataset/dataset/tr3_cropped/data/1.png" width="50%" height="50%"><img src="../train/manual_check/img/unet/predict_img1_LovaszLoss_ep10.png" width="50%" height="50%">

所以最后我们初步再对现有的loss function每个再循环20个epochs，观察其后续结果