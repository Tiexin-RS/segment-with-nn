import tensorflow as tf
import numpy as np
with tf.profiler.experimental.Profile('/home/Tiexin-RS/profiles'):
    with tf.device('gpu'):
        list_data = np.load('/home/Tiexin-RS/code/workspace/wjz/segment-with-nn/serving/load_test/locust_tfserving/a.npy')
        payload = {"inputs": {'input_1': list_data.tolist()}}
        tensor_data = tf.constant(list_data)

