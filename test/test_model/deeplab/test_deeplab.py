# coding=utf-8
''' test case for deeplab 
'''
import tensorflow as tf

from segelectri.model.deeplab import Deeplab


class TestDeeplabLayers(tf.test.TestCase):

    def test_xception_with_real_input(self):
        fake_input = tf.random.uniform((1, 512, 512, 3),
                                       minval=0,
                                       maxval=1,
                                       dtype=tf.float32)

        deeplab = Deeplab(num_classes=10)
        output = deeplab(fake_input)

        self.assertAllEqual(output.shape, [1, 512, 512, 10])
