# coding=utf-8
''' layer impl used for deeplab model 
'''
import tensorflow as tf
from tensorflow import keras


class SpatialPyramidPooling(tf.keras.layers.Layer):
    """Implements the Atrous Spatial Pyramid Pooling.
    Reference:
        [Rethinking Atrous Convolution for Semantic Image Segmentation](
          https://arxiv.org/pdf/1706.05587.pdf)
    """

    def __init__(self,
                 output_channels,
                 dilation_rates,
                 pool_kernel_size=None,
                 batchnorm_momentum=0.99,
                 batchnorm_epsilon=0.001,
                 activation='relu',
                 dropout=0.5,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 interpolation='bilinear',
                 **kwargs):
        """Initializes `SpatialPyramidPooling`.
        Args:
            output_channels: Number of channels produced by SpatialPyramidPooling.
            dilation_rates: A list of integers for parallel dilated conv.
            pool_kernel_size: A list of integers or None. If None, global average
                pooling is applied, otherwise an average pooling of pool_kernel_size
                is applied.
            batchnorm_momentum: A float for the momentum in BatchNorm. Defaults to
                0.99.
            batchnorm_epsilon: A float for the epsilon value in BatchNorm. Defaults to
                0.001.
            activation: A `str` for type of activation to be used. Defaults to 'relu'.
            dropout: A float for the dropout rate before output. Defaults to 0.5.
            kernel_initializer: Kernel initializer for conv layers. Defaults to
                `glorot_uniform`.
            kernel_regularizer: Kernel regularizer for conv layers. Defaults to None.
            interpolation: The interpolation method for upsampling. Defaults to
                `bilinear`.
            **kwargs: Other keyword arguments for the layer.
        """
        super(SpatialPyramidPooling, self).__init__(**kwargs)

        self.output_channels = output_channels
        self.dilation_rates = dilation_rates
        self.batchnorm_momentum = batchnorm_momentum
        self.batchnorm_epsilon = batchnorm_epsilon
        self.activation = activation
        self.dropout = dropout
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.interpolation = interpolation
        self.input_spec = tf.keras.layers.InputSpec(ndim=4)
        self.pool_kernel_size = pool_kernel_size

    def build(self, input_shape):
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        self.aspp_layers = []

        if tf.keras.backend.image_data_format() == 'channels_last':
            bn_axis = -1
        else:
            bn_axis = 1

        conv_sequential = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.output_channels,
                                   kernel_size=(1, 1),
                                   kernel_initializer=self.kernel_initializer,
                                   kernel_regularizer=self.kernel_regularizer,
                                   use_bias=False),
            tf.keras.layers.BatchNormalization(axis=bn_axis,
                                               momentum=self.batchnorm_momentum,
                                               epsilon=self.batchnorm_epsilon),
            tf.keras.layers.Activation(self.activation)
        ])
        self.aspp_layers.append(conv_sequential)

        for dilation_rate in self.dilation_rates:
            conv_sequential = tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    filters=self.output_channels,
                    kernel_size=(3, 3),
                    padding='same',
                    kernel_regularizer=self.kernel_regularizer,
                    kernel_initializer=self.kernel_initializer,
                    dilation_rate=dilation_rate,
                    use_bias=False),
                tf.keras.layers.BatchNormalization(
                    axis=bn_axis,
                    momentum=self.batchnorm_momentum,
                    epsilon=self.batchnorm_epsilon),
                tf.keras.layers.Activation(self.activation)
            ])
            self.aspp_layers.append(conv_sequential)

        if self.pool_kernel_size is None:
            pool_sequential = tf.keras.Sequential([
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Reshape((1, 1, channels))
            ])
        else:
            pool_sequential = tf.keras.Sequential(
                [tf.keras.layers.AveragePooling2D(self.pool_kernel_size)])

        pool_sequential.add(
            tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    filters=self.output_channels,
                    kernel_size=(1, 1),
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    use_bias=False),
                tf.keras.layers.BatchNormalization(
                    axis=bn_axis,
                    momentum=self.batchnorm_momentum,
                    epsilon=self.batchnorm_epsilon),
                tf.keras.layers.Activation(self.activation),
                tf.keras.layers.experimental.preprocessing.Resizing(
                    height,
                    width,
                    interpolation=self.interpolation,
                    dtype=tf.float32)
            ]))

        self.aspp_layers.append(pool_sequential)

        self.projection = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.output_channels,
                                   kernel_size=(1, 1),
                                   kernel_initializer=self.kernel_initializer,
                                   kernel_regularizer=self.kernel_regularizer,
                                   use_bias=False),
            tf.keras.layers.BatchNormalization(axis=bn_axis,
                                               momentum=self.batchnorm_momentum,
                                               epsilon=self.batchnorm_epsilon),
            tf.keras.layers.Activation(self.activation),
            tf.keras.layers.Dropout(rate=self.dropout)
        ])

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        result = []
        for layer in self.aspp_layers:
            result.append(
                tf.cast(layer(inputs, training=training), inputs.dtype))
        result = tf.concat(result, axis=-1)
        result = self.projection(result, training=training)
        return result

    def get_config(self):
        config = {
            'output_channels':
                self.output_channels,
            'dilation_rates':
                self.dilation_rates,
            'pool_kernel_size':
                self.pool_kernel_size,
            'batchnorm_momentum':
                self.batchnorm_momentum,
            'batchnorm_epsilon':
                self.batchnorm_epsilon,
            'activation':
                self.activation,
            'dropout':
                self.dropout,
            'kernel_initializer':
                tf.keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer':
                tf.keras.regularizers.serialize(self.kernel_regularizer),
            'interpolation':
                self.interpolation,
        }
        base_config = super(SpatialPyramidPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Decoder(keras.layers.Layer):
    """Implmentation of Decdoer for DeeplabV3+
    """

    def __init__(self, num_classes=10, upsample_factor=16, **kwargs):
        """Initializes `Decoder`.
        Args:
            num_class: `int` output channels suggests the classes for prediction
            upsample_factor: `int` division between original input image and aspp layer output
            **kwargs: Other keyword arguments for the layer
        """
        super(Decoder, self).__init__(**kwargs)

        self.num_classes = num_classes

        if not isinstance(upsample_factor, list):
            upsample_factor = (upsample_factor, upsample_factor)
        self.upsample_factor = upsample_factor

    def build(self, input_shape):
        low_feats_shape, high_feats_shape = input_shape

        high_feats_unsample_rate = (low_feats_shape[1] // high_feats_shape[1],
                                    low_feats_shape[2] // high_feats_shape[2])
        self.high_feats_upsample = keras.layers.UpSampling2D(
            size=high_feats_unsample_rate, interpolation='bilinear')
        self.low_feats_conv = keras.layers.Conv2D(filters=high_feats_shape[-1],
                                                  kernel_size=(1, 1))
        outputs_unsample_rates = (self.upsample_factor[0] //
                                  high_feats_unsample_rate[0],
                                  self.upsample_factor[1] //
                                  high_feats_unsample_rate[1])
        self.upsample_seq = keras.Sequential([
            keras.layers.Conv2D(filters=self.num_classes,
                                kernel_size=(3, 3),
                                padding='same'),
            keras.layers.UpSampling2D(size=outputs_unsample_rates,
                                      interpolation='bilinear'),
            keras.layers.Activation('softmax',
                                    dtype='float32',
                                    name='predictions')
        ])

    def call(self, inputs, **kwargs):
        """ call inputs with list as [low_feats, high_feats]
        """
        low_feats, high_feats = inputs
        x = tf.concat([
            tf.cast(self.low_feats_conv(low_feats), dtype=tf.float32),
            self.high_feats_upsample(high_feats)
        ],
                      axis=-1)
        outputs = self.upsample_seq(x)
        return outputs

    def get_config(self):
        config = {
            'num_classes': self.num_classes,
            'upsample_factor': self.upsample_factor,
        }
        base_config = super(Decoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
