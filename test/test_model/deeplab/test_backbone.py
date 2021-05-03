# coding=utf-8
''' test case for deeplab layers
'''
import tensorflow as tf

from segelectri.model.deeplab.backbone import Backbone


class TestDeeplabLayers(tf.test.TestCase):

    def test_backbone_with_real_input(self):
        fake_input = tf.keras.Input((224, 224, 3),
                                    dtype=tf.float32,
                                    batch_size=2)
        backbone = Backbone()
        outputs, skip = backbone(fake_input)
        self.assertAllEqual(outputs.shape, [2, 14, 14, 576])
        self.assertAllEqual(skip.shape, [2, 56, 56, 144])


if __name__ == '__main__':
    tf.test.main()
