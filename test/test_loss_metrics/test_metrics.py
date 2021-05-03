# coding=utf-8
''' test case for metrics
'''
import tensorflow as tf
from segelectri.loss_metrics.metrics import MeanIou


class TestMetrics(tf.test.TestCase):

    def test_meaniou(self):
        y_true = tf.random.uniform((2, 512, 512, 4),
                                   minval=0,
                                   maxval=3,
                                   dtype=tf.int64)
        y_pred = tf.random.uniform((2, 512, 512, 4),
                                   minval=0,
                                   maxval=1,
                                   dtype=tf.float32)
        iou = MeanIou()
        metrics = iou(y_true, y_pred)
        self.assertAllEqual(metrics.shape, ())


if __name__ == '__main__':
    tf.test.main()
