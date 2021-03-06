# coding=utf-8
''' test case for deeplab layers
'''
import tensorflow as tf
from segelectri.model.deeplab.mobilenet.mobilenet import Mobilenet


class TestDeeplabLayers(tf.test.TestCase):
    def test_mobilenet_with_input_layer(self):
        fake_input = tf.keras.Input((1024, 1024, 3), dtype=tf.float32)
        mobile = Mobilenet()
        outputs, skip = mobile(fake_input)
        self.assertAllEqual(outputs.shape, [None, 64, 64, 576])
        self.assertAllEqual(skip.shape, [None, 256, 256, 144])

    def test_mobilenet_with_real_input(self):
        fake_input = tf.keras.Input((1024, 1024, 3), dtype=tf.float32, batch_size=2)
        mobile = Mobilenet()
        outputs, skip = mobile(fake_input)
        mobile_serial = mobile.get_config()
        print(mobile_serial)
        self.assertAllEqual(outputs.shape, [2, 64, 64, 576])
        self.assertAllEqual(skip.shape, [2, 256, 256, 144])


if __name__ == '__main__':
    tf.test.main()
