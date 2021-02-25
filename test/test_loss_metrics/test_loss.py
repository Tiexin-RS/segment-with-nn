# coding=utf-8
''' test case for loss
'''
import tensorflow as tf
from segelectri.loss_metrics.loss import FocalLoss, LovaszLoss, DiceLoss, BoundaryLoss


class TestLoss(tf.test.TestCase):
    def test_focal_loss(self):
        y_true = tf.random.uniform((2, 512, 512),
                                   minval=0,
                                   maxval=3,
                                   dtype=tf.int64)
        y_pred = tf.random.uniform((2, 512, 512, 3),
                                   minval=0,
                                   maxval=1,
                                   dtype=tf.float32)
        focall = FocalLoss()
        loss = focall(y_true, y_pred)
        self.assertAllEqual(loss.shape, ())

    def test_lovasz_loss(self):
        y_true = tf.random.uniform((2, 512, 512),
                                   minval=0,
                                   maxval=3,
                                   dtype=tf.int64)
        y_pred = tf.random.uniform((2, 512, 512, 3),
                                   minval=0,
                                   maxval=1,
                                   dtype=tf.float32)
        lovaszl = LovaszLoss()
        loss = lovaszl(y_true, y_pred)
        self.assertAllEqual(loss.shape, ())

    def test_cross_entropy_loss(self):
        y_true = tf.random.uniform((2, 512, 512),
                                   minval=0,
                                   maxval=3,
                                   dtype=tf.int64)
        y_pred = tf.random.uniform((2, 512, 512, 3),
                                   minval=0,
                                   maxval=1,
                                   dtype=tf.float32)
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = scce(y_true, y_pred)
        self.assertAllEqual(loss.shape, ())

    def test_dice_loss(self):
        y_true = tf.random.uniform((2, 512, 512),
                                   minval=0,
                                   maxval=3,
                                   dtype=tf.int64)
        y_pred = tf.random.uniform((2, 512, 512, 3),
                                   minval=0,
                                   maxval=1,
                                   dtype=tf.float32)
        dicel = DiceLoss()
        loss = dicel(y_true, y_pred)
        self.assertAllEqual(loss.shape, ())

    def test_boundary_loss(self):
        y_true = tf.random.uniform((2, 512, 512),
                                   minval=0,
                                   maxval=3,
                                   dtype=tf.int64)
        y_pred = tf.random.uniform((2, 512, 512, 3),
                                   minval=0,
                                   maxval=1,
                                   dtype=tf.float32)
        boundaryl = BoundaryLoss()
        loss = boundaryl(y_true, y_pred)
        self.assertAllEqual(loss.shape, ())
