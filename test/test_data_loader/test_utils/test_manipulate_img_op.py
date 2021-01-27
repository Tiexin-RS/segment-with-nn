import unittest
import numpy as np
import tensorflow as tf

from segelectri.data_loader.utils.manipulate_img_op import generate_crop_boxes, get_available_stuff, split_img_op


class TestManipulateImgOp(unittest.TestCase):

    def test_get_available_stuff(self):
        self.assertEqual([1], get_available_stuff(1024, 1, 512))
        self.assertEqual([1, 513], get_available_stuff(1026, 1, 512))
        self.assertEqual([64, 1088, 2112], get_available_stuff(3328, 64, 1024))

    def test_generate_crop_boxes(self):
        num_box, boxes = generate_crop_boxes([1027, 1027], [1, 1], [512, 512])
        self.assertCountEqual([[1, 1, 513, 513], [513, 1, 1025, 513],
                               [1, 513, 513, 1025], [513, 513, 1025, 1025]],
                              boxes)
        self.assertEqual(4, num_box)

    def test_split_img_op(self):
        with tf.device('cpu'):
            fake_original_img = tf.random.uniform((3328, 3328, 3),
                                                  minval=0,
                                                  maxval=255,
                                                  dtype=tf.dtypes.int32)
            fake_mask_img = tf.random.uniform((3328, 3328, 3),
                                              minval=0,
                                              maxval=255,
                                              dtype=tf.dtypes.int32)

            outputs = split_img_op(fake_original_img,
                                   fake_mask_img,
                                   crop_size=(1024, 1024))
            shape = np.array(outputs).shape
            self.assertEqual(shape, (9, 2, 1024, 1024, 3))


if __name__ == "__main__":
    unittest.main()