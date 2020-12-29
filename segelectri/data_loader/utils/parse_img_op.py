# coding=utf-8
from re import match
import tensorflow as tf


def parse_img_and_mask(origin_img_path: tf.Tensor, mask_img_path: tf.Tensor):
    """parse img and mask according to path that provided

    Args:
        origin_img_path (tf.Tensor): origin image path
        mask_img_path (tf.Tensor): mask image path
    """
    origin_img = tf.io.decode_png(tf.io.read_file(origin_img_path), channels=3)
    mask_img = tf.io.decode_png(tf.io.read_file(mask_img_path), channels=1)

    assert origin_img.shape[:2] == mask_img.shape[:2], \
        'mask image and origin image shape doesn\'t match'

    return (origin_img, mask_img)