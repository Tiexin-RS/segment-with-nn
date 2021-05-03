import tensorflow as tf


class Mobilenet(tf.keras.layers.Layer):

    def __init__(self,
                 backbone='mobilenetv2',
                 mobile_weights='imagenet',
                 layer_names=['block_3_expand_relu', 'block_13_expand_relu'],
                 **kwargs):
        super(Mobilenet, self).__init__(**kwargs)
        self.backbone = backbone
        self.mobile_weights = mobile_weights
        self.layer_names = layer_names

    def build(self, input_shape):
        backbone_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape[1:],
            weights=self.mobile_weights,
            include_top=False)
        layers = [
            backbone_model.get_layer(name).output for name in self.layer_names
        ]
        self.layer = tf.keras.Model(inputs=backbone_model.input, outputs=layers)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        [skip, output] = self.layer(inputs, training=training)
        return output, skip

    def get_config(self):
        config = {'backbone': self.backbone}
        base_config = super(Mobilenet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
