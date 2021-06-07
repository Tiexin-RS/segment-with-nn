import logging
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from segelectri.loss_metrics.loss import FocalLoss, LovaszLoss, DiceLoss, BoundaryLoss
from segelectri.loss_metrics.metrics import MeanIou
import time


def read_image(file_name, resize=True):
    img = tf.io.read_file(filename=file_name)
    img = tf.io.decode_image(img)
    if resize:
        img = tf.image.resize(img, [224, 224])
    return img


def save_image(save_data, save_path):
    plt.imshow(save_data)
    plt.colorbar()
    plt.savefig(save_path, dpi=300)


def get_label(label_path, resize=True):
    label_data = read_image(label_path, resize)
    label_data = label_data[:, :, 1]
    return label_data


def get_pred(seg_model, data_path, resize=True):
    ori_data = read_image(data_path, resize)
    if resize:
        ori_data = tf.cast(tf.reshape(ori_data,
                                      (-1, 224, 224, 3)), tf.float32) / 255.0
    else:
        ori_data = tf.cast(tf.reshape(ori_data,
                                      (-1, 1024, 1024, 3)), tf.float32) / 255.0
    
    with tf.device('/cpu:0'):
        begin = time.perf_counter()
        pred_data = seg_model.predict(ori_data)
        end = time.perf_counter()
        print('cpu total time:%.5f' % (end-begin))
    # pred_data = seg_model.predict(ori_data)
    # begin = time.perf_counter()
    # pred_data = seg_model.predict(ori_data)
    # end = time.perf_counter()
    # print('gpu total time:%.5f' % (end-begin))
    pred_data = tf.argmax(pred_data, axis=-1)
    if resize:
        pred_data = tf.reshape(pred_data, (224, 224))
    else:
        pred_data = tf.reshape(pred_data, (1024, 1024))
    return pred_data


def show_label(label_path, label_save_path, resize=True):
    label_data = get_label(label_path, resize)
    save_image(label_data, label_save_path)


def show_pred(seg_model, data_path, data_save_path, resize=True):
    pred_data = get_pred(seg_model, data_path, resize)
    save_image(pred_data, data_save_path)


def show_meaniou(seg_model, resize=True):
    m = MeanIou(num_classes=4)
    label_p = Path('/opt/dataset/tr3_cropped/label/')
    original_p = Path('/opt/dataset/tr3_cropped/data/')
    label_list = list(sorted(label_p.glob("*.png")))
    original_list = list(sorted(original_p.glob("*.png")))
    for label_path, original_path in zip(label_list, original_list):
        label_data = get_label(str(label_path), resize)
        pred_data = get_pred(seg_model, str(original_path), resize)
        label_data = tf.cast(label_data, tf.int64)
        pred_data = tf.cast(pred_data, tf.int64)
        label_data = tf.one_hot(label_data, 4)
        pred_data = tf.one_hot(pred_data, 4)
        m.update_state(pred_data, label_data)
        logging.info('iou is %s', m.result().numpy())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    label_path = '/opt/dataset/tr2_cropped/label/125.png'
    label_save_path = 'img/label/img_125_resize.png'
    data_path = '/opt/dataset/tr2_cropped/data/125.png'
    data_save_path = 'img/unet/tr2/cross/img125_ep20.png'
    model_path = '/opt/segelectri/train/train_unet/normal_function_model/clustered_model/1'
    # '/opt/segelectri/train/train_unet/normal_function_model/pruned_pb_model/1'
    seg_model = tf.keras.models.load_model(filepath=model_path,
                                           custom_objects={
                                               'MeanIou': MeanIou,
                                               'FocalLoss': FocalLoss,
                                               'LovaszLoss': LovaszLoss,
                                               'DiceLoss': DiceLoss,
                                               'BoundaryLoss': BoundaryLoss
                                           })
    show_pred(seg_model, data_path, data_save_path,resize = False)
    # seg_model.summary()