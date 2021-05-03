# Deeplab问题

## 代码情况
选择采用mobilenet的预训练权重，对backbone进行freeze训练10epochs, 然后unfreeze训练20epochs, 损失函数采用lovasz, batch_size=32, 更详细情况可见训练代码和train/train_deeplab/exp/43/config，另外我为了unfreeze就把routine.run这个方法里边改了一下

## resize情况
将图片resize为(224, 224)，下图为原图和resize图的对比，可以看到基本上没啥差别：

<img src="../train/manual_check/img/label/label_img_1.png" width="50%" height="50%"><img src="../train/manual_check/img/label/deeplab/../label_img_1_resize.png" width="50%" height="50%">

## Metrics问题
这是我刚刚训练的一个，下图为tensorboard metrics:

<img src="../train/manual_check/img/tensorboard/lovasz_metrics.png" width="50%" height="50%">

可见最终metrics大约为0.55

在train/manual_check/ans.txt中有对458张图片进行推理优化的metrics情况，从上到下排列，最终mean = 0.26999716068686375，
相关代码见manual_deeplab.py

## 手动推理图片
效果奇差，iou = 0.19552675基本吻合

<img src="../train/manual_check/img/deeplab/tr3/lovasz/img1_ep10+20_batch32.png" width="50%" height="50%">

