# coding=utf-8
''' layer impl used for deeplab model 
'''
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, ZeroPadding2D)


class Conv2dSame(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 stride=1,
                 kernel_size=3,
                 rate=1,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 **kwargs):

        super(Conv2dSame, self).__init__(**kwargs)

        self.filters = filters
        self.stride = stride
        self.kernel_size = kernel_size
        self.rate = rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        if self.stride == 1:
            depth_padding = 'same'
        else:
            depth_padding = 'valid'

        self.conv_sequential = tf.keras.Sequential([
            Conv2D(filters=self.filters,
                   kernel_size=(self.kernel_size, self.kernel_size),
                   kernel_regularizer=self.kernel_regularizer,
                   kernel_initializer=self.kernel_initializer,
                   strides=(self.stride, self.stride),
                   padding=depth_padding,
                   use_bias=False,
                   dilation_rate=(self.rate, self.rate))
        ])

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        # 计算padding的数量，hw是否需要收缩
        if self.stride != 1:
            kernel_size_effective = self.kernel_size + (self.kernel_size -
                                                        1) * (self.rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            inputs = ZeroPadding2D((pad_beg, pad_end))(inputs)

        result = tf.cast(self.conv_sequential(inputs, training=training),
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
            'kernel_initializer':
            tf.keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer':
            tf.keras.regularizers.serialize(self.kernel_regularizer),
        }
        base_config = super(Conv2dSame, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
