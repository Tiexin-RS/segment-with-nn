# coding=utf-8
import os
import unittest
import tensorflow as tf

from segelectri.train.train_routine import TrainRoutine
from segelectri.data_loader.tr_ds.tr_ds import get_tr_ds
from segelectri.model.deeplab import Deeplab
from segelectri.loss_metrics.loss import FocalLoss


class TestTrainRoutine(unittest.TestCase):

    def test_ensure_dir_exists(self):
        TrainRoutine._ensure_dir_exists('demo')

        os.mkdir('ex')
        TrainRoutine._ensure_dir_exists('ex')

        self.assertTrue(os.path.exists('demo'))
        self.assertTrue(os.path.exists('ex'))

        os.removedirs('demo')
        os.removedirs('ex')

    def test_run(self):
        train_data = tf.random.uniform((10, 224, 224, 3),
                                       minval=0,
                                       maxval=1,
                                       dtype=tf.float32)
        train_label = tf.random.uniform((10, 1000),
                                        minval=0,
                                        maxval=1,
                                        dtype=tf.float32)

        ds = tf.data.Dataset.from_tensor_slices((train_data, train_label))

        model = tf.keras.applications.DenseNet121(input_shape=(224, 224, 3),
                                                  weights=None)
        model.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.BinaryCrossentropy())
        routine = TrainRoutine(ds=ds, model=model)
        routine.run(exp_dir='exp', epochs=1, batch_size=2)

        self.assertTrue(os.path.exists('exp/config/exp_config.json'))

    def diff_model(self):
        with tf.device('cpu'):
            ds = get_tr_ds(
                original_pattern='/opt/dataset/tr2_cropped/data/*.png',
                mask_pattern='/opt/dataset/tr2_cropped/label/*.png',
                batch_size=2)

            def reshape_fn(d, l):
                d = tf.cast(tf.reshape(d,
                                        (-1, 1024, 1024, 3)), tf.float32) / 255.0
                l = tf.reshape(l, (-1, 1024, 1024, 3))
                return d, l

            ds = ds.map(reshape_fn)

        model = Deeplab(num_classes=3)
        # model.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            # loss=FocalLoss()
        )
        #   metrics=[tf.keras.metrics.MeanIoU(num_classes=3)])
        routine = TrainRoutine(ds=ds, model=model)
        routine.run(exp_dir='exp/01', epochs=1)

        self.assertTrue(os.path.exists('exp/01/config/exp_config.json'))


if __name__ == '__main__':
    unittest.main()
