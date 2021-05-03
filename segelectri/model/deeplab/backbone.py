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
        self.backbone = BACKBONE_MODEL_MAPPER[self.backbone_name]()

    def call(self, inputs):
        outputs, skip_feats = self.backbone(inputs)
        return outputs, skip_feats

    def get_config(self):
        config = {'backbone_name': self.backbone_name}
        base_config = super(Backbone, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
