import tensorflow as tf
import numpy as np

from typing import List, Tuple


def get_available_stuff(dim_length: int, useless_padding: int, box_length: int):
    """helper function of get available stuff for generate_crop_boxes

    Args:
        length (int): total length for height or width
        useless_padding (int): useless padding along length
        box_length (int): box length along this dim
    """
    curr_idx = useless_padding
    available_stuff = []
    while curr_idx + box_length + useless_padding <= dim_length:
        available_stuff.append(curr_idx)
        curr_idx += box_length
    return available_stuff


def generate_crop_boxes(\
    img_shape: Tuple[int, int],
    useless_padding: Tuple[int, int],
    box_shape: Tuple[int, int]):
    """generate crop boxes

    Args:
        img_shape (Tuple): input img shape
        useless_padding (int): useless padding around the img
        box_shape (Tuple): box shape
    
    Returns:
        num_box (int): number of boxes
        boxes (List shape: [num_box, 4]): coordinates of boxes
    """
    useless_padding = [0 if u <= 0 else u for u in useless_padding]
    width_available = get_available_stuff(img_shape[0], useless_padding[0],
                                          box_shape[0])
    height_available = get_available_stuff(img_shape[1], useless_padding[1],
                                           box_shape[1])

    boxes = []
    box_width = box_shape[0]
    box_height = box_shape[1]
    for w in width_available:
        for h in height_available:
            boxes.append([h, w, h + box_height, w + box_width])

    num_box = len(height_available) * len(width_available)
    return num_box, boxes


def normalize_boxes(img_shape: Tuple[int, int], boxes: List[float]):
    return boxes / np.array([
        img_shape[1] - 1, img_shape[0] - 1, img_shape[1] - 1, img_shape[0] - 1
    ])


def split_img_op(original_img: tf.Tensor, mask_img: tf.Tensor,
                 crop_size: Tuple[int, int]):
    """split original img and mask img into `crop_size`
    
        if shape of original one and mask one diffs, crop around the bigger one.

    Args:
        original_img (tf.Tensor): raw content of original img
        mask_img (tf.Tensor): raw content of mask img
        crop_size (Tuple[int, int]): crop size you want

    Returns:
        (num_of_box, 2): lots of pair of original crop and mask crop
    """
    original_shape, mask_shape = np.array(original_img.shape), np.array(
        mask_img.shape)
    useless_padding = (original_shape - mask_shape) / 2
    original_num_box, original_boxes = generate_crop_boxes(
        original_shape, useless_padding, crop_size)
    original_boxes = normalize_boxes(img_shape=original_shape,
                                     boxes=original_boxes)
    mask_num_box, mask_boxes = generate_crop_boxes(mask_shape, -useless_padding,
                                                   crop_size)
    mask_boxes = normalize_boxes(img_shape=mask_shape, boxes=mask_boxes)
    assert original_num_box == mask_num_box, 'not equal mask img boxes and original one'

    outputs = []
    box_inds = tf.zeros(original_num_box, dtype=tf.dtypes.int32)
    for img, boxes in [(original_img, original_boxes), (mask_img, mask_boxes)]:
        img = tf.expand_dims(img, axis=0)
        outputs.append(
            tf.image.crop_and_resize(image=img,
                                     boxes=boxes,
                                     box_indices=box_inds,
                                     crop_size=crop_size))

    outputs = [(o, m) for o, m in zip(outputs[0], outputs[1])]
    outputs = tf.convert_to_tensor(outputs, dtype=tf.uint8)
    return outputs
