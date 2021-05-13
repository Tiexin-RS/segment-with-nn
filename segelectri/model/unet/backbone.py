from traceback import print_tb
import tensorflow as tf

BACKBONE_MODEL_MAPPER = {
    'mobilenet': tf.keras.applications.MobileNetV2,
}

class Backbone(tf.keras.layers.Layer):

    def __init__(self,
                 backbone='mobilenet',
                 mobile_weights=None,
                 layer_names=['block_1_expand_relu', 'block_2_project_BN', 'block_4_project_BN', 'block_6_project_BN'],
                 **kwargs):
        super(Backbone, self).__init__(**kwargs)
        self.backbone = backbone
        self.mobile_weights = mobile_weights
        self.layer_names = layer_names

    def build(self, input_shape):
        backbone_model = BACKBONE_MODEL_MAPPER[self.backbone](
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

        skips = self.layer(inputs, training=training)
        return skips

    def get_config(self):
        config = {'backbone': self.backbone, 'weights': self.mobile_weights, 'layer_names': self.layer_names}
        base_config = super(Backbone, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    fake_input = tf.keras.Input((1024, 1024, 3), dtype=tf.float32, batch_size=2)
    backbone = Backbone()
    skips = backbone(fake_input)
    shapes = [
        [2, 512, 512, 96],
        [2, 256, 256, 24],
        [2, 128, 128, 32],
        [2, 64, 64, 64]
    ]
    for i in range(len(skips)):
        print(skips[i].shape)