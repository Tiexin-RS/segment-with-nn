# coding=utf-8
''' test case for loss
'''
import tensorflow as tf
from segelectri.loss_metrics.loss import FocalLoss, LovaszLoss, DiceLoss, BoundaryLoss


class TestLoss(tf.test.TestCase):

    def setUp(self):
        self.y_true = tf.random.uniform((2, 512, 512, 4),
                                        minval=0,
                                        maxval=3,
                                        dtype=tf.int64)
        self.y_pred = tf.random.uniform((2, 512, 512, 4),
                                        minval=0,
                                        maxval=1,
                                        dtype=tf.float32)

    def test_focal_loss(self):
        focall = FocalLoss()
        loss = focall(self.y_true, self.y_pred)
        self.assertAllEqual(loss.shape, ())

    def test_lovasz_loss(self):
        lovaszl = LovaszLoss()
        loss = lovaszl(self.y_true, self.y_pred)
        self.assertAllEqual(loss.shape, ())

    def test_cross_entropy_loss(self):
        scce = tf.keras.losses.BinaryCrossentropy()
        loss = scce(self.y_true, self.y_pred)
        self.assertAllEqual(loss.shape, ())

    def test_dice_loss(self):
        dicel = DiceLoss()
        loss = dicel(self.y_true, self.y_pred)
        self.assertAllEqual(loss.shape, ())

    def test_boundary_loss(self):
        boundaryl = BoundaryLoss()
        loss = boundaryl(self.y_true, self.y_pred)
        self.assertAllEqual(loss.shape, ())


if __name__ == '__main__':
    tf.test.main()
