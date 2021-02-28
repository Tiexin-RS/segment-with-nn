# coding=utf-8
from tensorflow import keras

from .deeplab_layers import SpatialPyramidPooling, Decoder
from .xception.xception import Xception


class Deeplab(keras.Model):
    """impl for DeeplabV3+
    """

    def __init__(self,
                 dilation_rates=[1, 2, 4, 6, 12],
                 num_classes=3,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.dilation_rates = dilation_rates
        self.num_classes = num_classes

        self.xception = Xception()
        self.aspp = SpatialPyramidPooling(output_channels=256,
                                          dilation_rates=self.dilation_rates)
        self.decoder = Decoder(num_classes=num_classes, upsample_factor=8)

    def call(self, inputs):
        outputs, skip_feats = self.xception(inputs)
        high_feats = self.aspp(outputs)
        outputs = self.decoder([skip_feats, high_feats])

        return outputs
