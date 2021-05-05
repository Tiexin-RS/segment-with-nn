import tensorflow as tf
import matplotlib.pyplot as plt
from segelectri.loss_metrics.loss import FocalLoss, LovaszLoss, DiceLoss, BoundaryLoss

def read_image(file_name):
    img = tf.io.read_file(filename=file_name)
    img = tf.image.decode_jpeg(img, channels=0)
    return img


if __name__ == "__main__":
    seg_model = tf.keras.models.load_model("../train_unet/exp/31/saved_model",custom_objects={'LovaszLoss': LovaszLoss})
    original_data = read_image("/opt/dataset/tr3_cropped/data/1.png")
    original_data = tf.cast(tf.reshape(original_data, (-1, 1024, 1024, 3)), tf.float32) / 255.0
    predict_data = seg_model(original_data)
    predict_argmax = tf.argmax(predict_data, axis=-1)
    predict_redata = tf.reshape(predict_argmax, (1024, 1024))
    plt.imshow(predict_redata)
    plt.savefig('img/unet/predict_img1_LovaszLoss_ep20_preload.png', dpi=300)