from locust import HttpUser, task
from locust import User
import tensorflow as tf
from locust.contrib.fasthttp import FastHttpUser


def read_image(file_name, resize=True):
    img = tf.io.read_file(filename=file_name)
    img = tf.io.decode_image(img)
    if resize:
        img = tf.image.resize(img, [224, 224])
    return img


# class QuickstartUser(HttpUser):

#     # wait_time = between(1, 2.5)

#     @task
#     def tf_serving_test(self):
#         data_path = '/home/Tiexin-RS/dataset/tr3_cropped/data/1.png'
#         ori_data = read_image(data_path, False)
#         ori_data = tf.cast(tf.reshape(ori_data, (-1, 1024, 1024, 3)), tf.float32)
#         # ori_data = tf.random.uniform((1, 1024, 1024, 3),
#         #                               minval=0,
#         #                               maxval=255,
#         #                               dtype=tf.float32)
#         data = ori_data.numpy()
#         payload = {"inputs": {'input_1': data.tolist()}}
#         self.client.post("v1/models/deeplab_52_unfreeze:predict", json=payload)


class QuickstartUser(FastHttpUser):

    # wait_time = between(1, 2.5)
    def on_start(self):
        data_path = '/home/Tiexin-RS/dataset/tr3_cropped/data/1.png'
        ori_data = read_image(data_path, False)
        ori_data = tf.cast(tf.reshape(ori_data, (-1, 1024, 1024, 3)),
                           tf.float32)
        data = ori_data.numpy()
        self.payload = {"inputs": {'input_1': data.tolist()}}

    @task
    def tf_serving_test(self):
        self.client.request(method='POST',
                            path="v1/models/deeplab_52_unfreeze:predict",
                            json=self.payload)
