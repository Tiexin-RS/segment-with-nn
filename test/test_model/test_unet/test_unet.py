# coding=utf-8
''' test case for unet
'''
import tensorflow as tf
from segelectri.model.unet import unet_t

class TestUnet(tf.test.TestCase):
    
    def test_weights(self):
        fake_input = tf.keras.Input((512, 512, 1), dtype=tf.float32)
        unet = unet_t.Unet()
        output = unet(fake_input)
        print(output.shape)
        # test output.shape
        self.assertAllEqual(output.shape, [None, 512,512, 10])
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


if __name__ == "__main__":
    tf.test.main()
