# coding=utf-8
''' test case for deeplab layers
'''
import tensorflow as tf

from segelectri.model.deeplab.deeplab_layers import SpatialPyramidPooling


class TestDeeplabLayers(tf.test.TestCase):

    def test_aspp(self):
        fake_input = tf.keras.Input((64, 64, 128), dtype=tf.float32)
        aspp = SpatialPyramidPooling(output_channels=128,
                                     dilation_rates=[1, 6, 12])
        outputs = aspp(fake_input)
        self.assertAllEqual(outputs.shape, [None, 64, 64, 128])
