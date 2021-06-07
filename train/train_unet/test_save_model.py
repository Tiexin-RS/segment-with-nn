import logging
import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from segelectri.loss_metrics.loss import FocalLoss, LovaszLoss, DiceLoss, BoundaryLoss
from segelectri.loss_metrics.metrics import MeanIou

@tf.function
def serve(model,inputs):
    return model(inputs)

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
save_path = '/opt/segelectri/train/train_unet/tmp/saved_model/1'
ret = serve(seg_model,inputs)
print(type(ret))