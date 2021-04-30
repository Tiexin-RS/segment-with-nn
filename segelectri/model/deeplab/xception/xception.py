# coding=utf-8
''' layer impl used for deeplab model 
'''
import tensorflow as tf
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D)
from segelectri.model.deeplab.xception.xception_block import XceptionBlock


class Xception(tf.keras.layers.Layer):
    """Implements Xception.
    Reference:
        [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](
          https://arxiv.org/pdf/1802.02611.pdf)
    """
    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 batchnorm_momentum=0.99,
                 batchnorm_epsilon=0.001,
                 activation='relu',
                 **kwargs):

        super(Xception, self).__init__(**kwargs)

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

        # pre
        # 1024,1024,3 -> 512,512,32 -> 512,512,64
        self.pre_layers = []
        pre_filters = [32, 64]
        pre_strides = [2, 1]
        for i in range(len(pre_filters)):
            if i==0:
                conv_sequential = tf.keras.Sequential([
                    Conv2D(filters=pre_filters[i],
                        kernel_size=(3, 3),
                        padding='same',
                        kernel_regularizer=self.kernel_regularizer,
                        kernel_initializer=self.kernel_initializer,
                        strides=(pre_strides[i], pre_strides[i]),
                        use_bias=False,
                        input_shape=input_shape[1:]),
                ])
            else:
                conv_sequential = tf.keras.Sequential([
                    Conv2D(filters=pre_filters[i],
                        kernel_size=(3, 3),
                        padding='same',
                        kernel_regularizer=self.kernel_regularizer,
                        kernel_initializer=self.kernel_initializer,
                        strides=(pre_strides[i], pre_strides[i]),
                        use_bias=False),
                ])
            conv_sequential.add(tf.keras.Sequential([
                BatchNormalization(axis=bn_axis,
                                   momentum=self.batchnorm_momentum,
                                   epsilon=self.batchnorm_epsilon),
                Activation(self.activation)
            ]))
            self.pre_layers.append(conv_sequential)

        # 512,512,64 -> 256,256,128
        self.entry_layer1 = XceptionBlock(depth_list=[128, 128, 128],
                                          skip_connection_type='conv',
                                          stride=2,
                                          depth_activation=False)
        # 256,256,128 -> 128,128,256
        # skip = 256,256,256                
        self.entry_layer2 = XceptionBlock(depth_list=[256, 256, 256],
                                    skip_connection_type='conv',
                                    stride=2,
                                    depth_activation=False,
                                    return_skip=True)
        # 128,128,256 -> 128,128,728
        self.entry_layer3 = XceptionBlock(depth_list=[728, 728, 728],
                                          skip_connection_type='conv',
                                          stride=1,
                                          depth_activation=False)
        # middle flow
        # 128,128,728 -> 128,128,728
        self.middle_layers = []
        for i in range(16):
            self.middle_layers.append(
                XceptionBlock(depth_list=[728, 728, 728],
                              skip_connection_type='sum',
                              stride=1,
                              rate=1,
                              depth_activation=False))
        
        # exit flow
        # 64,64,728 -> 64,64,1024
        self.exit_layer1 = XceptionBlock(depth_list=[728, 1024, 1024],
                                         skip_connection_type='conv',
                                         stride=1,
                                         rate=1,
                                         depth_activation=False)
        # 64,64,1024 -> 64,64,2048
        self.exit_layer2 = XceptionBlock(depth_list=[1536, 1536, 2048],
                                         skip_connection_type='none',
                                         stride=1,
                                         rate=2,
                                         depth_activation=True)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        result = inputs
        for layer in self.pre_layers:
            result = tf.cast(layer(result, training=training), inputs.dtype)
        
        result = tf.cast(self.entry_layer1(result, training=training),
                         inputs.dtype)
        result, skip1 = self.entry_layer2(result, training=training)
        result = tf.cast(self.entry_layer3(result, training=training),
                         inputs.dtype)
        # middle flow
        for layer in self.middle_layers:
            result = tf.cast(layer(result, training=training), inputs.dtype)
        
       # exit flow
        result = tf.cast(self.exit_layer1(result, training=training),
                         inputs.dtype)
        result = tf.cast(self.exit_layer2(result, training=training),
                         inputs.dtype)
        return result, skip1
         
    def get_config(self):
        config = {
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
        base_config = super(Xception, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
