# coding=utf-8
import os
import unittest
import tensorflow as tf

from segelectri.train.train_routine import TrainRoutine
from segelectri.data_loader.tr_ds.tr_ds import get_tr_ds
from segelectri.model.unet.unet_t import Unet, get_cluster_unet, get_prunable_unet
from segelectri.model.unet.unet_t import get_unet
from segelectri.loss_metrics.loss import FocalLoss, LovaszLoss, DiceLoss, BoundaryLoss
from segelectri.loss_metrics.metrics import MeanIou
import tempfile
import tensorflow_model_optimization as tfmot


if __name__ == '__main__':
    # Try to add mix precision to accelerate train speed
    # policy = tf.keras.mixed_precision.Policy('mixed_float16')
    # tf.keras.mixed_precision.set_global_policy(policy)
    
    with tf.device('/cpu:0'):
        ds = get_tr_ds(
            original_pattern='/opt/dataset/tr2_cropped/data/*.png',
            mask_pattern='/opt/dataset/tr2_cropped/label/*.png',
            batch_size=1)

        def reshape_fn(d, l):
            d = tf.cast(tf.reshape(d, (-1, 1024, 1024, 3)), tf.float32) / 255.0
            l = l[:,:,:,1]
            l = tf.one_hot(l, 4)
            l = tf.reshape(l, (-1, 1024, 1024, 4))
            return d, l

        ds = ds.map(reshape_fn)

    loss_list = [tf.keras.losses.BinaryCrossentropy(),FocalLoss(),LovaszLoss(),DiceLoss(), BoundaryLoss()]
    expdir_list = ['exp/11', 'exp/12', 'exp/23', 'exp/24', 'exp/25']
    
    model = get_prunable_unet(depth=3)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = LovaszLoss(),
        metrics = [MeanIou(num_classes=4)]
    )
    log_dir = tempfile.mkdtemp(dir='./prunable_model')
    cbs = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        # Log sparsity and other metrics in Tensorboard.
        tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
    ]
    model.fit(ds,epochs = 20,callbacks= cbs)
    model.save('./prunable_model/saved_model')