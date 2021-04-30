# coding=utf-8
from tensorflow import keras

from .deeplab_layers import SpatialPyramidPooling, Decoder
from .xception.xception import Xception
from .mobilenet.mobilenet import Mobilenet

class Deeplab(keras.Model):
    """impl for DeeplabV3+
    """

    def __init__(self,
                 dilation_rates=[1, 2, 4, 6, 12],
                 num_classes=3,
                 backbone='mobilenet',
                 *args,
                 **kwargs):
        super(Deeplab, self).__init__(*args, **kwargs)

        self.dilation_rates = dilation_rates
        self.num_classes = num_classes
        if backbone=='mobilenet':
            self.backbone = Mobilenet()
        elif backbone=='xception':
            self.backbone = Xception()
        self.aspp = SpatialPyramidPooling(output_channels=256,
                                          dilation_rates=self.dilation_rates)
        self.decoder = Decoder(num_classes=num_classes, upsample_factor=16)

    def call(self, inputs):
        outputs, skip_feats = self.backbone(inputs)
        high_feats = self.aspp(outputs)
        outputs = self.decoder([skip_feats, high_feats])
        return outputs

    def get_config(self):
        backbone_config = self.backbone.get_config()
        aspp_config = self.aspp.get_config()
        decoder_config = self.decoder.get_config()
        # base_config = super(Deeplab, self).get_config()

        config = {
            'backbone': backbone_config,
            'aspp': aspp_config,
            'decoder_config': decoder_config
        }
        # config.update(base_config)

        return config
