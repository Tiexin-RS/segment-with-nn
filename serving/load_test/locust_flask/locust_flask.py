from locust import HttpUser, task, between
import tensorflow as tf

def read_image(file_name, resize=True):
        img = tf.io.read_file(filename=file_name)
        img = tf.io.decode_image(img)
        if resize:
            img = tf.image.resize(img, [224, 224])
        return img


class QuickstartUser(HttpUser):
    
    # wait_time = between(1, 2.5)

    @task
    def tf_flask_test(self):
        data_path = '/opt/dataset/tr3_cropped/data/1.png'
        ori_data = read_image(data_path, False)
        ori_data = tf.cast(tf.reshape(ori_data, (-1, 1024, 1024, 3)), tf.float32)
        data = ori_data.numpy()
        payload = {'file': data.tolist()}
        self.client.post("predict", json=payload)
        # image = open(data_path, "rb").read()
        # payload = {"file": image}
        # self.client.post("predict", files=payload)
