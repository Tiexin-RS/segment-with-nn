# coding=utf-8
import tensorflow as tf
import tensorflow_io as tfio


def decode_image(path: tf.Tensor):
    """decode fn for tiff, png, jpg, bmp, giff format

    Args:
        path (tf.Tensor): path for this image
    """
    decode_fns = [tf.image.decode_image, tfio.experimental.image.decode_tiff]
    raw_content = tf.io.read_file(path)
    for fn in decode_fns:
        try:
            return fn(raw_content)
        except Exception:
            continue

    raise RuntimeError('failed to decode image {}'.format(path.numpy()))


def parse_img_and_mask(origin_img_path: tf.Tensor, mask_img_path: tf.Tensor):
    """parse img and mask according to path that provided

    Args:
        origin_img_path (tf.Tensor): origin image path
        mask_img_path (tf.Tensor): mask image path
    """
    origin_img = decode_image(origin_img_path)
    mask_img = decode_image(mask_img_path)[:, :, :3]

    # assert origin_img.shape[:2] == mask_img.shape[:2], \
    #    'mask image and origin image shape doesn\'t match'

    return origin_img, mask_img


def save_img(img: tf.Tensor, img_path: str):
    """save image

    Args:
        img (tf.Tensor): img content
        img_path (str): img path

    """
    return tf.io.write_file(img_path, tf.image.encode_png(img))
