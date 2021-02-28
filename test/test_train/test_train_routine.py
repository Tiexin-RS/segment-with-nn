# coding=utf-8
import os
import unittest
import tensorflow as tf

from segelectri.train.train_routine import TrainRoutine


class TestTrainRoutine(unittest.TestCase):

    def test_ensure_dir_exists(self):
        os.removedirs('demo')
        os.removedirs('ex')

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
