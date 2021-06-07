import psutil as ps
import time
import tensorflow as tf
import numpy as np
from segelectri.loss_metrics.loss import FocalLoss, LovaszLoss, DiceLoss, BoundaryLoss
from segelectri.loss_metrics.metrics import MeanIou

def running(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        print("func run time {}".format(end_time - start_time))
    return wrapper

@running
def cpu_predict(ori_data, model):
    # with tf.profiler.experimental.Profile('/opt/segelectri/serving/sdk/profile-sdk'):
    return model.predict(ori_data)

@running
def gpu_predict(ori_data, model):
    # with tf.profiler.experimental.Profile('/opt/segelectri/serving/sdk/profile-sdk'):
    return model.predict(ori_data)

if __name__ == "__main__":
    # tf.config.optimizer.set_jit(True)
    with tf.device('/cpu:0'):
        model_path = '/opt/segelectri/train/train_unet/normal_function_model/clustered_model/1'
        # model_path = '/opt/segelectri/train/train_deeplab/exp/55_unfreeze/saved_model/1'
        ori_data = np.load('/opt/segelectri/serving/tf_serving/a.npy')
        ori_data = tf.constant(ori_data)
        model = tf.keras.models.load_model(filepath=model_path,
                                        custom_objects={
                                            'MeanIou': MeanIou,
                                            'FocalLoss': FocalLoss,
                                            'LovaszLoss': LovaszLoss,
                                            'DiceLoss': DiceLoss,
                                            'BoundaryLoss': BoundaryLoss
                                        })
        # model.predict(ori_data)
        # gpu_predict(ori_data, model)
        model.summary()
