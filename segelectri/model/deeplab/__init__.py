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
                 *args,
                 **kwargs):
        super(Deeplab, self).__init__(*args, **kwargs)

        self.dilation_rates = dilation_rates
        self.num_classes = num_classes

        # self.xception = Xception()
        self.mobilenet = Mobilenet()
        self.aspp = SpatialPyramidPooling(output_channels=256,
                                          dilation_rates=self.dilation_rates)
        self.decoder = Decoder(num_classes=num_classes, upsample_factor=16)

    def call(self, inputs):
        # outputs, skip_feats = self.xception(inputs)
        outputs, skip_feats = self.mobilenet(inputs)
        high_feats = self.aspp(outputs)
        outputs = self.decoder([skip_feats, high_feats])

        return outputs

    def get_config(self):
        # xception_config = self.xception.get_config()
        mobilenet_config = self.mobilenet.get_config()
        aspp_config = self.aspp.get_config()
        decoder_config = self.decoder.get_config()
        # base_config = super(Deeplab, self).get_config()

        config = {
            # 'xception': xception_config,
            'mobilenet': mobilenet_config,
            'aspp': aspp_config,
            'decoder_config': decoder_config
        }
        # config.update(base_config)

        return config
