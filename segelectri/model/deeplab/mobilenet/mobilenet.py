import tensorflow as tf
from segelectri.model.deeplab.backbones import BACKBONES

class Mobilenet(tf.keras.layers.Layer):
    def __init__(self, backbone='mobilenetv2', **kwargs):
        super(Mobilenet, self).__init__(**kwargs)
        self.backbone = backbone

    def _get_backbone_feature(self, feature: str, input_shape) -> tf.keras.Model:
        input_layer = tf.keras.Input(shape=input_shape[1:])
        backbone_model = BACKBONES[self.backbone]['model'](input_tensor=input_layer, weights=None, include_top=False)
        output_layer = backbone_model.get_layer(BACKBONES[self.backbone][feature]).output
        return tf.keras.Model(inputs=input_layer, outputs=output_layer)

    def build(self, input_shape):
        self.backbone_feature_1 = self._get_backbone_feature('feature_1', input_shape)
        self.backbone_feature_2 = self._get_backbone_feature('feature_2', input_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        
        skip = tf.cast(self.backbone_feature_1(inputs, training=training), inputs.dtype)
        output = tf.cast(self.backbone_feature_2(inputs, training=training), inputs.dtype)
        return output, skip

    def get_config(self):
        config = {
            'backbone':
            self.backbone
        }
        base_config = super(Mobilenet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



