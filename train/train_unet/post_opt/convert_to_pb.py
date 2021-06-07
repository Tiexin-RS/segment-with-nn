import logging
import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow._api.v2 import sparse
from tensorflow.python.keras.backend import dtype
from tensorflow.python.saved_model.utils_impl import get_tensor_from_tensor_info
from segelectri.loss_metrics.loss import FocalLoss, LovaszLoss, DiceLoss, BoundaryLoss
from segelectri.loss_metrics.metrics import MeanIou
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework.errors_impl import InvalidArgumentError

# model_path = '/opt/segelectri/train/train_unet/normal_function_model/pruned_pb_model/1'
# model = tf.keras.models.load_model(filepath=model_path,
#                                     custom_objects={
#                                         'MeanIou': MeanIou,
#                                         'FocalLoss': FocalLoss,
#                                         'LovaszLoss': LovaszLoss,
#                                         'DiceLoss': DiceLoss,
#                                         'BoundaryLoss': BoundaryLoss
#                                     })
model_path = './tmp/tmp5oynqapi.h5'
model = tf.keras.models.load_model(filepath=model_path,
                                    custom_objects={
                                        'MeanIou': MeanIou,
                                        'FocalLoss': FocalLoss,
                                        'LovaszLoss': LovaszLoss,
                                        'DiceLoss': DiceLoss,
                                        'BoundaryLoss': BoundaryLoss
                                    })
save_path = '/opt/segelectri/train/train_unet/normal_function_model/pruned_pb_model/1'
tf.keras.Model.save(model,save_path)