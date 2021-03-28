# coding=utf-8
''' test case for deeplab 
'''
import tensorflow as tf

from segelectri.model.deeplab import Deeplab
from segelectri.model.unet import unet_t

class TestDeeplabLayers(tf.test.TestCase):

    def test_xception_with_real_input(self):
        fake_input = tf.random.uniform((1, 1024, 1024, 3),
                                       minval=0,
                                       maxval=1,
                                       dtype=tf.float32)

        deeplab = Deeplab(num_classes=10)
        output = deeplab(fake_input)

        self.assertAllEqual(output.shape, [1, 1024, 1024, 10])
    
    def test_weights(self):
        fake_input = tf.keras.Input((512, 512, 1), dtype=tf.float32)
        unet = unet_t.Unet()
        output = unet(fake_input)
        print(output.shape)
        # test output.shape
        self.assertAllEqual(output.shape, [None, 512, 512, 10])
        # test weights.shape
        w = [(3, 3, 1, 64), (64, ), (3, 3, 64, 64), (64, ), (3, 3, 64, 128),
             (128, ), (3, 3, 128, 128), (128, ), (3, 3, 128, 256), (256, ),
             (3, 3, 256, 256), (256, ), (3, 3, 256, 512), (512, ),
             (3, 3, 512, 512), (512, ), (3, 3, 512, 1024), (1024, ),
             (3, 3, 1024, 1024), (1024, ), (2, 2, 1024, 512), (512, ),
             (3, 3, 1024, 512), (512, ), (3, 3, 512, 512), (512, ),
             (2, 2, 512, 256), (256, ), (3, 3, 512, 256), (256, ),
             (3, 3, 256, 256), (256, ), (2, 2, 256, 128), (128, ),
             (3, 3, 256, 128), (128, ), (3, 3, 128, 128), (128, ),
             (2, 2, 128, 64), (64, ), (3, 3, 128, 64), (64, ), (3, 3, 64, 64),
             (64, ), (3, 3, 64, 10), (10, )]
        for weight,rw in zip(unet.weights,w):
            self.assertAllEqual(weight.shape,rw)

if __name__ == '__main__':
    tf.test.main()
