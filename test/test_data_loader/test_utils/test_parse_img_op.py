# coding=utf-8
import unittest
import logging
import tensorflow as tf

from segelectri.data_loader.utils.parse_img_op import parse_img_and_mask


class TestParseImgOp(unittest.TestCase):
    def test_true(self):
        self.assertTrue(True)

    def parse_sample_file(self):
        origin_img_path = tf.constant('/opt/dataset/ccf-train/2.png')
        mask_img_path = tf.constant('/opt/dataset/ccf-train/2_class.png')
        images = parse_img_and_mask(origin_img_path=origin_img_path,
                                    mask_img_path=mask_img_path)
        logging.info('shape of img is {}'.format(images[0].shape))
        return images


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
