# coding=utf-8
''' layer impl used for deeplab model 
'''
import tensorflow as tf
from tensorflow.keras.layers import (Activation, add, BatchNormalization, Concatenate,
                                    Conv2D, DepthwiseConv2D, Dropout,
                                    GlobalAveragePooling2D, Input, Lambda, ZeroPadding2D)
                                    
from segelectri.model.deeplab.xception.sepconv_bn import SepconvBn
from segelectri.model.deeplab.xception.conv2d_same import Conv2dSame

class XceptionBlock(tf.keras.layers.Layer):
    def __init__(   self,
                    depth_list, 
                    skip_connection_type, 
                    stride,
                    rate=1, 
                    depth_activation=False, 
                    return_skip=False,
                    batchnorm_momentum=0.99,
                    batchnorm_epsilon=0.001,
                    **kwargs):

        super(XceptionBlock, self).__init__(**kwargs)

        self.depth_list = depth_list
        self.skip_connection_type = skip_connection_type
        self.stride = stride
        self.rate = rate
        self.depth_activation = depth_activation
        self.return_skip = return_skip
        self.batchnorm_momentum = batchnorm_momentum
        self.batchnorm_epsilon = batchnorm_epsilon

    def build(self, input_shape):
        if tf.keras.backend.image_data_format() == 'channels_last':
            bn_axis = -1
        else:
            bn_axis = 1

        self.sepconv_layers = []
        for i in range(3):    
            sepconv = tf.keras.Sequential([
                SepconvBn(  filters=self.depth_list[i],
                            stride=self.stride if i == 2 else 1,
                            rate=self.rate,
                            depth_activation=self.depth_activation)
            ])
            self.sepconv_layers.append(sepconv)

        self.conv_connect = tf.keras.Sequential([
            Conv2dSame( filters=self.depth_list[-1], 
                        kernel_size=1,
                        stride=self.stride),
            BatchNormalization( axis=bn_axis,
                                momentum=self.batchnorm_momentum,
                                epsilon=self.batchnorm_epsilon)
        ])

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        residual = inputs
        index = 0
        for layer in self.sepconv_layers:
            residual = tf.cast(layer(residual, training=training), inputs.dtype)
            if index==1:
                skip = residual
            index = index + 1

        if self.skip_connection_type == 'conv':
            shortcut = self.conv_connect(inputs, training=training)
            outputs = add([residual, shortcut])
        elif self.skip_connection_type == 'sum':
            outputs = add([residual, inputs])
        elif self.skip_connection_type == 'none':
            outputs = residual
        
        if self.return_skip:
            return outputs, skip
        else:
            return outputs     

    def get_config(self):
        config = {
            'depth_list':
                self.depth_list,
            'skip_connection_type':
                self.skip_connection_type,            
            'stride':
                self.stride,
            'rate':
                self.rate,
            'depth_activation':
                self.depth_activation,      
            'return_skip':
                self.return_skip,            
            'batchnorm_momentum':
                self.batchnorm_momentum,
            'batchnorm_epsilon':
                self.batchnorm_epsilon,
        }
        base_config = super(XceptionBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
