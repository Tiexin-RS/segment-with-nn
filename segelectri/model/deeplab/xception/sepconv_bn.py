# coding=utf-8
''' layer impl used for deeplab model 
'''
import tensorflow as tf
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     DepthwiseConv2D, ZeroPadding2D)


class SepconvBn(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 stride=1,
                 kernel_size=3,
                 rate=1,
                 depth_activation=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 batchnorm_momentum=0.99,
                 batchnorm_epsilon=0.001,
                 activation='relu',
                 **kwargs):

        super(SepconvBn, self).__init__(**kwargs)

        self.filters = filters
        self.stride = stride
        self.kernel_size = kernel_size
        self.rate = rate
        self.depth_activation = depth_activation
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.batchnorm_momentum = batchnorm_momentum
        self.batchnorm_epsilon = batchnorm_epsilon
        self.activation = activation

    def build(self, input_shape):
        if tf.keras.backend.image_data_format() == 'channels_last':
            bn_axis = -1
        else:
            bn_axis = 1

        if self.stride == 1:
            depth_padding = 'same'
        else:
            depth_padding = 'valid'

        self.activate = Activation(self.activation)

        # 分离卷积，3x3采用膨胀卷积
        self.expand_conv = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=(self.kernel_size, self.kernel_size),
                            strides=(self.stride, self.stride),
                            dilation_rate=(self.rate, self.rate),
                            padding=depth_padding,
                            use_bias=False,
                            depthwise_initializer=self.kernel_initializer,
                            depthwise_regularizer=self.kernel_regularizer),
            BatchNormalization(axis=bn_axis,
                               momentum=self.batchnorm_momentum,
                               epsilon=self.batchnorm_epsilon)
        ])
        if self.depth_activation:
            self.expand_conv.add(
                tf.keras.Sequential([Activation(self.activation)]))

        # 1x1卷积，进行压缩
        self.compress_conv = tf.keras.Sequential([
            Conv2D(filters=self.filters,
                   kernel_size=(1, 1),
                   kernel_regularizer=self.kernel_regularizer,
                   kernel_initializer=self.kernel_initializer,
                   strides=(1, 1),
                   padding='same',
                   use_bias=False),
            BatchNormalization(axis=bn_axis,
                               momentum=self.batchnorm_momentum,
                               epsilon=self.batchnorm_epsilon)
        ])
        if self.depth_activation:
            self.compress_conv.add(
                tf.keras.Sequential([Activation(self.activation)]))

        if self.stride != 1:
            kernel_size_effective = self.kernel_size + (self.kernel_size - 1) * (self.rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            self.zero_padding = ZeroPadding2D((pad_beg, pad_end))

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        # 计算padding的数量，hw是否需要收缩
        if self.stride != 1:
            inputs = self.zero_padding(inputs)

        # 如果需要激活函数
        if not self.depth_activation:
            inputs = tf.cast(self.activate(inputs, training=training),
                             inputs.dtype)
        result = tf.cast(self.expand_conv(inputs, training=training),
                         inputs.dtype)
        result = tf.cast(self.compress_conv(result, training=training),
                         inputs.dtype)

        return result

    def get_config(self):
        config = {
            'filters':
                self.filters,
            'stride':
                self.stride,
            'kernel_size':
                self.kernel_size,
            'rate':
                self.rate,
            'depth_activation':
                self.depth_activation,
            'kernel_initializer':
                tf.keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer':
                tf.keras.regularizers.serialize(self.kernel_regularizer),
            'batchnorm_momentum':
                self.batchnorm_momentum,
            'batchnorm_epsilon':
                self.batchnorm_epsilon,
            'activation':
                self.activation,
        }
        base_config = super(SepconvBn, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
