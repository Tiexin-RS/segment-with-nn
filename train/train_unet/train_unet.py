# coding=utf-8
import os
import unittest
import tensorflow as tf
from segelectri.data_loader.utils.manipulate_img_op import split_img_op
from segelectri.train.train_routine import TrainRoutine
from segelectri.data_loader.tr_ds.tr_ds import get_tr_ds
from segelectri.model.unet.unet_t import Unet, get_cluster_unet, get_prunable_unet,get_ordinary_unet
from segelectri.model.unet.unet_t import get_unet
from segelectri.loss_metrics.loss import FocalLoss, LovaszLoss, DiceLoss, BoundaryLoss
from segelectri.loss_metrics.metrics import MeanIou
import tempfile


if __name__ == '__main__':
    # Try to add mix precision to accelerate train speed
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    with tf.device('/cpu:0'):
        ds = get_tr_ds(
            original_pattern='/opt/dataset/tr3_cropped/data/*.png',
            mask_pattern='/opt/dataset/tr3_cropped/label/*.png',
            batch_size=16)

        def reshape_fn(d, l):
            d = tf.cast(tf.reshape(d, (-1, 1024, 1024, 3)), tf.float32) / 255.0
            l = l[:,:,:,1]
            l = tf.one_hot(l, 4)
            l = tf.reshape(l, (-1, 1024, 1024, 4))
            l = tf.image.resize(l, [224, 224]) # resize
            return d, l

        ds = ds.map(reshape_fn)

    loss_list = [tf.keras.losses.BinaryCrossentropy(),FocalLoss(),LovaszLoss(),DiceLoss(), BoundaryLoss()]
    expdir_list = ['exp/11', 'exp/12', 'exp/23', 'exp/24', 'exp/25']
    
    # sub-class
    model = Unet(depth=4,num_classes = 4,pre_encoder=True,resize = True)#preload_encoder=False,quantized=False
    # output = model(fake_input)
    # tfmot.quantization
    # quant_aware_model = get_unet(depth = 3)
    # model = tfmot.quantization.keras.quantize_apply(quant_aware_model)
    # model = get_ordinary_unet(depth = 4,resize = True,mixed_float16 = True)
    # model = get_prunable_unet(depth = 3)
    # model = get_cluster_unet(depth=3)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = LovaszLoss(),
        metrics = [MeanIou(num_classes=4)]
    )

    # model.summary()
    routine = TrainRoutine(ds=ds, model = model)
    routine.run(exp_dir='exp/63',epochs = 20) # 50:CPU without quant_aware 51:CPU with quant_aware 
    #'60':no pretrain,no float 16
    #'61':no pretrain,float 16
    #'62':pre-train,no float 16
    #'63':pre-train,float 16
