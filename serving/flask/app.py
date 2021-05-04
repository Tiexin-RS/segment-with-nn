from flask import Flask, request, render_template, jsonify
import os
import uuid
import io
from PIL import Image
import tensorflow as tf
from segelectri.loss_metrics.loss import FocalLoss, LovaszLoss, DiceLoss, BoundaryLoss
from segelectri.loss_metrics.metrics import MeanIou
from train.manual_infer.infer_deeplab import show_pred

app = Flask(__name__)
app.config["SECRET_KEY"] = 'TPmi4aLWRbyVq8zu9v82dWYW1'

model = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in set(['png', 'PNG'])


def upload_image(rest_ful=False):
    if rest_ful:
        image = request.files['file'].read()
        image = Image.open(io.BytesIO(image))
        basepath = os.path.dirname(__file__)
        new_filename = 'upload_img/' + str(uuid.uuid1()) + '.png'
        upload_path = os.path.join(basepath, './static/', new_filename)
        image.save(upload_path)
        return new_filename
    else:
        f = request.files.get('file')
        if not (f and allowed_file(f.filename)):
            return ''
        else:
            basepath = os.path.dirname(__file__)
            ext = os.path.splitext(f.filename)[1]
            new_filename = 'upload_img/' + str(uuid.uuid1()) + ext
            upload_path = os.path.join(basepath, './static/', new_filename)
            f.save(upload_path)
            return new_filename


def load_model():
    """Load the pre-trained model, you can use your model just as easily.
    """
    global seg_model
    model_path = '/opt/segelectri/train/train_deeplab/exp/43_unfreeze/saved_model'
    seg_model = tf.keras.models.load_model(filepath=model_path,
                                           custom_objects={
                                               'MeanIou': MeanIou,
                                               'FocalLoss': FocalLoss,
                                               'LovaszLoss': LovaszLoss,
                                               'DiceLoss': DiceLoss,
                                               'BoundaryLoss': BoundaryLoss
                                           })


@app.route('/predict', methods=['POST'])
def predict():
    data = {"success": False}
    if request.method == 'POST':
        upload_path = upload_image(rest_ful=True)
        if upload_path != '':
            data_path = './static/' + upload_path
            data_save_path = 'pred_img/' + str(uuid.uuid1()) + '.png'
            show_pred(seg_model, data_path, './static/' + data_save_path)
            data["success"] = True
            data['data_save_path'] = data_save_path
        return jsonify(data)


@app.route('/', methods=['POST', 'GET'])
def seg():
    if request.method == 'POST':
        upload_path = upload_image()
        if upload_path != '':
            data_path = './static/' + upload_path
            data_save_path = 'pred_img/' + str(uuid.uuid1()) + '.png'
            show_pred(seg_model, data_path, './static/' + data_save_path)
            return render_template('index.html',
                                   upload_path=upload_path,
                                   data_save_path=data_save_path)
    return render_template('index.html')


if __name__ == '__main__':
    load_model()
    app.run()
