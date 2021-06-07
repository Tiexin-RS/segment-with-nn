# coding=utf-8
import tensorflow as tf
from .xception.xception import Xception
from .mobilenet.mobilenet import Mobilenet

BACKBONE_MODEL_MAPPER = {
    'mobilenet': Mobilenet,
    'xception': Xception,
}


class Backbone(tf.keras.layers.Layer):

    def __init__(self, backbone_name='mobilenet', **kwargs):
        super(Backbone, self).__init__(**kwargs)

        self.backbone_name = backbone_name

    def build(self, input_shape):
        self.resize_and_rescale = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Resizing(224, 224),
            tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        ])
        self.backbone = BACKBONE_MODEL_MAPPER[self.backbone_name]()

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        inputs = self.resize_and_rescale(inputs)
        outputs, skip_feats = self.backbone(inputs)
        return outputs, skip_feats

    def get_config(self):
        config = {'backbone_name': self.backbone_name}
        base_config = super(Backbone, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
