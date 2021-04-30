# coding=utf-8
''' test case for deeplab layers
'''
import tensorflow as tf
from tensorflow.python.keras.backend import dtype

from segelectri.model.deeplab.deeplab_layers import Decoder, SpatialPyramidPooling


class TestDeeplabLayers(tf.test.TestCase):

    def test_aspp_with_input_layer(self):
        fake_input = tf.keras.Input((64, 64, 576), dtype=tf.float32)
        aspp = SpatialPyramidPooling(output_channels=128,
                                     dilation_rates=[1, 6, 12])
        outputs = aspp(fake_input)
        self.assertAllEqual(outputs.shape, [None, 64, 64, 128])

    def test_aspp_with_real_input(self):
        fake_input = tf.random.uniform((1, 64, 64, 576),
                                       minval=0,
                                       maxval=1,
                                       dtype=tf.float32)
        aspp = SpatialPyramidPooling(output_channels=128,
                                     dilation_rates=[1, 6, 12])
        outputs = aspp(fake_input)
        self.assertAllEqual(outputs.shape, [1, 64, 64, 128])

    def test_decoder(self):
        fake_low_feats = tf.keras.Input((256, 256, 144), dtype=tf.float32)
        fake_high_feats = tf.keras.Input((64, 64, 128), dtype=tf.float32)

        decoder = Decoder(num_classes=6, upsample_factor=16)
        outputs = decoder([fake_low_feats, fake_high_feats])
        self.assertAllEqual(outputs.shape, [None, 1024, 1024, 6])

        decoder_serial = decoder.get_config()
        self.assertTrue(decoder_serial['num_classes'], 6)


if __name__ == '__main__':
    tf.test.main()
