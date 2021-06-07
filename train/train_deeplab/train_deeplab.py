# coding=utf-8
import tensorflow as tf
from segelectri.train.train_routine import TrainRoutine
from segelectri.data_loader.tr_ds.tr_ds import get_tr_ds
from segelectri.model.deeplab import Deeplab
from segelectri.loss_metrics.loss import FocalLoss, LovaszLoss, DiceLoss, BoundaryLoss
from segelectri.loss_metrics.metrics import MeanIou


def model_optimize(xla=False, mix_prec=True):
    '''Try to enable xla to accelerate train speed
       Try to add mix precision to accelerate train speed 
    '''
    if xla:
        tf.config.optimizer.set_jit(True)
    if mix_prec:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)


def train(loss_list,
          freeze_expdir_list,
          unfreeze_expdir_list,
          batch_size=32,
          resize=True,
          freeze_epochs=10,
          unfreeze_epochs=5):
    with tf.device('cpu'):
        ds = get_tr_ds(original_pattern='/opt/dataset/tr3_cropped/data/*.png',
                       mask_pattern='/opt/dataset/tr3_cropped/label/*.png',
                       batch_size=batch_size)

        def reshape_fn(d, l):
            d = tf.cast(tf.reshape(d, (-1, 1024, 1024, 3)), tf.float32)
            l = l[:, :, :, 1]
            l = tf.one_hot(l, 4)
            l = tf.reshape(l, (-1, 1024, 1024, 4))
            if resize:
                l = tf.image.resize(l, [224, 224])
            return d, l

        ds = ds.map(reshape_fn)

    for i in range(len(loss_list)):
        model = Deeplab(num_classes=4)
        model.layers[0].trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=loss_list[i],
                      metrics=[MeanIou(num_classes=4)])
        routine = TrainRoutine(ds=ds, model=model)
        routine.run(exp_dir=freeze_expdir_list[i], epochs=freeze_epochs)

        model.layers[0].trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=loss_list[i],
                      metrics=[MeanIou(num_classes=4)])
        routine = TrainRoutine(ds=ds, model=model)
        routine.run(exp_dir=unfreeze_expdir_list[i], epochs=unfreeze_epochs)


if __name__ == '__main__':
    loss_list = [
        tf.keras.losses.BinaryCrossentropy(),
        FocalLoss(),
        LovaszLoss(),
        DiceLoss(),
        BoundaryLoss()
    ]
    freeze_expdir_list = [
        'exp/46_freeze', 'exp/47_freeze', 'exp/48_freeze', 'exp/49_freeze',
        'exp/50_freeze'
    ]
    unfreeze_expdir_list = [
        'exp/46_unfreeze', 'exp/47_unfreeze', 'exp/48_unfreeze', 'exp/49_unfreeze',
        'exp/50_unfreeze'
    ]
    model_optimize(xla=True, mix_prec=True)
    train(loss_list=loss_list,
          freeze_expdir_list=freeze_expdir_list,
          unfreeze_expdir_list=unfreeze_expdir_list,
          batch_size=32,
          resize=True,
          freeze_epochs=20,
          unfreeze_epochs=5)
