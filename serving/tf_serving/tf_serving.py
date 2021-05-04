import json
import requests
import tensorflow as tf


def read_image(file_name, resize=True):
    img = tf.io.read_file(filename=file_name)
    img = tf.io.decode_image(img)
    if resize:
        img = tf.image.resize(img, [224, 224])
    return img


def get_pred(data_path, url, resize=True):
    ori_data = read_image(data_path, resize)
    if resize:
        ori_data = tf.cast(tf.reshape(ori_data,
                                      (-1, 224, 224, 3)), tf.float32) / 255.0
    else:
        ori_data = tf.cast(tf.reshape(ori_data,
                                      (-1, 1024, 1024, 3)), tf.float32) / 255.0
    data = ori_data.numpy()
    payload = {"inputs": {'input_1': data.tolist()}}
    r = requests.post(url=url, json=payload)
    pred = json.loads(r.content.decode('utf-8'))
    json_data = json.dumps(pred)
    json_file = open('./pred.json', 'w')
    json_file.write(json_data)
    json_file.close()


if __name__ == "__main__":
    file_path = './static/img/1.png'
    url = 'http://localhost:8501/v1/models/deeplab_1:predict'
    get_pred(file_path, url, resize=False)
