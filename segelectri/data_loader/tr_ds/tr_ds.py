import glob
import os
import tensorflow as tf
import numpy as np

from segelectri.data_loader.utils.parse_img_op import parse_img_and_mask, save_img
from segelectri.data_loader.utils.manipulate_img_op import split_img_op


def generate_files(original_pattern: str, mask_pattern: str):
    """generator for original and mask img path

    Args:
        original_pattern (str): original path pattern
        mask_pattern (str): mask path pattern
    """

    def real_generator():
        original_img_paths = sorted(glob.glob(original_pattern))
        mask_img_paths = sorted(glob.glob(mask_pattern))
        for o, m in zip(original_img_paths, mask_img_paths):
            yield o, m

    return real_generator


def py_func_split_img_op(o, m):
    """py func wrapper for split img python kernel logic

    Args:
        o (tf.Tensor): original img tensor
        m (tf.Tensor): mask img tensor

    Returns:
        py_func: Operation wrapper for python kernel
    """
    wrapper_split_img_op = lambda o, m: split_img_op(o, m, (1024, 1024))
    return tf.py_function(func=wrapper_split_img_op, inp=[o, m], Tout=tf.uint8)


def py_func_parse_img_and_mask(o, m):
    """py func wrapper for parse original and mask img python kernel logic

    Args:
        o (tf.Tensor): original img path tensor
        m (tf.Tensor): mask img path tensor

    Returns:
        py_func: Operation wrapper for python kernel
    """

    return tf.py_function(func=parse_img_and_mask,
                          inp=[o, m],
                          Tout=(tf.uint8, tf.uint8))


def py_func_stack_into_one(o, m):
    """merge mask and original img into one tensor

    Args:
        o (tf.Tensor): original img tensor
        m (tf.Tensor): mask img tensor
    """

    def wrapper_stack_into_one(o, m):
        return tf.stack([o, m])

    return tf.py_function(func=wrapper_stack_into_one,
                          inp=[o, m],
                          Tout=tf.uint8)


def process_tr_data(original_pattern: str, mask_pattern: str,
                    processed_img_path: str):
    """process original img into (1024, 1024) crop

    Args:
        original_pattern (str): original img path pattern
        mask_pattern (str): mask img path pattern
        processed_img_path (str): path you want to store your processed img

    Returns:
        tf.data.Dataset: ds for process imgs
    """
    data_path = os.path.join(processed_img_path, 'data')
    label_path = os.path.join(processed_img_path, 'label')
    if not os.path.exists(processed_img_path):
        os.mkdir(processed_img_path)
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    generator = generate_files(original_pattern, mask_pattern)
    output_signature = (tf.TensorSpec(shape=(), dtype=tf.string),
                        tf.TensorSpec(shape=(), dtype=tf.string))
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)\
        .map(py_func_parse_img_and_mask, num_parallel_calls=tf.data.AUTOTUNE)\
        .map(py_func_split_img_op, num_parallel_calls=tf.data.AUTOTUNE)\
        .interleave(lambda x: tf.data.Dataset.from_tensor_slices(x), num_parallel_calls=tf.data.AUTOTUNE, cycle_length=4, block_length=16)

    print("Saving cropped image...")
    for idx, (o, m) in enumerate(ds):
        original_path = os.path.join(data_path, f'{idx}.png')
        mask_path = os.path.join(label_path, f'{idx}.png')
        save_img(o, original_path)
        save_img(m, mask_path)
    print("Saved...")
    return ds


def get_tr_ds(original_pattern: str,
              mask_pattern: str,
              batch_size: int = 32,
              buffer_size: int = 1024 * 1024 * 128):
    """get data input pipeline of data

    Args:
        original_pattern (str): original img path pattern
        mask_pattern (str): mask img path pattern
        batch_size (int): batch size
        buffer_size (int): buffer size to cache

    Returns:
        tf.data.Dataset: dataset of data
    """
    decode_image_fn = lambda o: tf.image.decode_image(tf.io.read_file(o))

    original_imgs = sorted(glob.glob(original_pattern))
    mask_imgs = sorted(glob.glob(mask_pattern))
    ds = tf.data.Dataset.from_tensor_slices((original_imgs, mask_imgs))\
        .map(lambda d, l: (decode_image_fn(d), decode_image_fn(l)), num_parallel_calls=tf.data.AUTOTUNE)
    if batch_size:
        ds = ds.batch(batch_size=batch_size)

    return ds


if __name__ == "__main__":
    # ds = process_tr_data('/opt/dataset/tr2/clip_*.png',
    #                      '/opt/dataset/tr2/clip_class_*.tif',
    #                      '/opt/dataset/tr2_cropped/')

    ds = process_tr_data('/opt/dataset/tr3/clip_*.bmp',
                         '/opt/dataset/tr3/class_clip_*.tif',
                         '/opt/dataset/tr3_cropped/')
