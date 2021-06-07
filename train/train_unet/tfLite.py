import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
import numpy as np
from segelectri.loss_metrics.loss import FocalLoss, LovaszLoss, DiceLoss, BoundaryLoss
from segelectri.loss_metrics.metrics import MeanIou
from segelectri.data_loader.tr_ds.tr_ds import get_tr_ds
from segelectri.data_loader.utils.parse_img_op import parse_img_and_mask, save_img
import pathlib
import glob
assert float(tf.__version__[:3]) >= 2.3

with tf.device('/cpu:0'):
        ds = get_tr_ds(
            original_pattern='/opt/dataset/tr2_cropped/data/*.png',
            mask_pattern='/opt/dataset/tr2_cropped/label/*.png',
            batch_size=1)

        def reshape_fn(d, l):
            d = tf.cast(tf.reshape(d, (-1, 1024, 1024, 3)), tf.float32) / 255.0
            l = l[:,:,:,1]
            l = tf.one_hot(l, 4)
            l = tf.reshape(l, (-1, 1024, 1024, 4))
            return d, l

        ds = ds.map(reshape_fn)

model_path = '/opt/segelectri/train/train_unet/normal_function_model/saved_model/1'
model = tf.keras.models.load_model(filepath=model_path,
                                        custom_objects={
                                            'MeanIou': MeanIou,
                                            'FocalLoss': FocalLoss,
                                            'LovaszLoss': LovaszLoss,
                                            'DiceLoss': DiceLoss,
                                            'BoundaryLoss': BoundaryLoss
                                        })

original_pattern='/opt/dataset/tr2_cropped/data/*.png'
mask_pattern='/opt/dataset/tr2_cropped/label/*.png'

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.float16]
tflite_model_quant = converter.convert()

tflite_models_dir = pathlib.Path("./tmp/unet_tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the quantized model:
tflite_model_quant_file = tflite_models_dir/"quanted_model.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)