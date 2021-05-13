from os import name
import tensorflow as tf
from tensorflow import keras
from .backbone import Backbone


class downsamp_conv(keras.layers.Layer):
    def __init__(self, filters_num=128,**kwargs):
        """Initialize 'downsamp_conv'
        Args:
            filters_num:num of filters in corresponding layer when downsampling
        """
        super(downsamp_conv, self).__init__(**kwargs)

        self.filters_num = filters_num

    def build(self, input_shape):
        self.shape = input_shape

        self.conv_seq = keras.Sequential([
            keras.layers.Conv2D(filters=self.filters_num,
                                kernel_size=(3, 3),
                                kernel_initializer='he_normal',
                                activation = 'relu',
                                padding='same'),
            keras.layers.BatchNormalization(axis = -1,
                                beta_initializer='zero',
                                gamma_initializer='one'),
            keras.layers.Conv2D(filters=self.filters_num,
                                kernel_size=(3, 3),
                                kernel_initializer='he_normal',
                                activation = 'relu',
                                padding='same'),
            keras.layers.BatchNormalization(axis = -1,
                                beta_initializer='zero',
                                gamma_initializer='one'),
        ])

    def call(self, inputs):
        output = self.conv_seq(inputs)
        return output

    def get_config(self):
        config = {
            'filter_num' : self.filters_num
        }
        base_config = super(downsamp_conv,self).get_config()
        return dict(list(base_config.items())+list(config.items()))


class upsamp_conv(keras.layers.Layer):
    def __init__(self, filters_num=128, **kwargs):
        """Initialize 'upsamp_conv'
        Args:
            filters_num:num of filter in corresponding layer
        """
        super(upsamp_conv, self).__init__(**kwargs)

        self.filters_num = filters_num

    def build(self, input_shape):
        self.shape = input_shape

        self.upsample_seq = keras.Sequential([
            keras.layers.Conv2D(filters=self.filters_num,
                                kernel_size=(3, 3),
                                kernel_initializer='he_normal',
                                activation = 'relu',
                                padding='same'),
            keras.layers.BatchNormalization(axis = -1,
                                beta_initializer='zero',
                                gamma_initializer='one'),
            keras.layers.Conv2D(filters=self.filters_num,
                                kernel_size=(3, 3),
                                kernel_initializer='he_normal',
                                activation = 'relu',
                                padding='same'),
            keras.layers.BatchNormalization(axis = -1,
                                beta_initializer='zero',
                                gamma_initializer='one'),
            keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            keras.layers.Conv2D(filters=self.filters_num / 2,
                                kernel_size=(2, 2),
                                kernel_initializer='he_normal',
                                activation = 'relu',
                                padding='same'),
            keras.layers.BatchNormalization(axis = -1,
                                beta_initializer='zero',
                                gamma_initializer='one')
        ])

    def call(self, inputs):
        output = self.upsample_seq(inputs)
        return output

    def get_config(self):
        config = {
            'filter_num' : self.filters_num
        }
        base_config = super(upsamp_conv,self).get_config()
        return dict(list(base_config.items())+list(config.items()))


class Unet(keras.Model):
    def __init__(self, min_kernel_num=64, num_classes=10,depth = 4,pre_encoder = False,
                layer_names=['block_1_expand_relu', 'block_2_project_BN', 'block_4_project_BN', 'block_6_project_BN'],**kwargs):
        """Initialize Unet
        Args:
            min_kernel_num:num of filters in the toppest layer
            num_classes:num of classes used to do segmetation
            depth:depth of unet,only under pre_encoder == False shall this param be valid
            pre_encoder:denote whether to load MobileNetV2 as encoder or not
            layer_names:denote layer_names when using pre_encoder
        """
        super(Unet, self).__init__(**kwargs)
        self.min_kernel_num = min_kernel_num // 1  #assure that is a int
        self.num_classes = num_classes
        self.depth = depth
        self.pre_encoder = pre_encoder
        self.layer_names = layer_names
        if self.pre_encoder:
            self.depth = 4 #depth has to be 4

        self.down_kernel_num_seq = [] # default 4 layers
        #根据层数生成
        for i in range(self.depth):
            self.down_kernel_num_seq.append(min_kernel_num*(2**i))
        self.up_kernel_num_seq = self.down_kernel_num_seq.copy() 
        self.up_kernel_num_seq.append(self.down_kernel_num_seq[-1]*2)
        self.up_kernel_num_seq = self.up_kernel_num_seq[:0:-1] # 反序list 
        # 并且减去一层最上层 交给output处理以进行语义分割

    def build(self, input_shape):
        self.shape = input_shape

        # use loop to wrap a list
        self.pre_encoder_layers = []
        self.down_conv_layers = []
        if self.pre_encoder:
           self.pre_encoder_layers = Backbone(layer_names = self.layer_names) 
        else:
            for k in self.down_kernel_num_seq:
                self.down_conv_layers.append(downsamp_conv(k))
        
        # use loop to wrap a list
        self.up_conv_layers = []
        
        for k in self.up_kernel_num_seq:
            self.up_conv_layers.append(upsamp_conv(k))

        self.output_seq = keras.Sequential([
            keras.layers.Conv2D(filters=self.down_kernel_num_seq[0],
                                kernel_size=(3, 3),
                                kernel_initializer='he_normal',
                                activation = 'relu',
                                padding='same'),
            keras.layers.BatchNormalization(axis = -1,
                                beta_initializer='zero',
                                gamma_initializer='one'),
            keras.layers.Conv2D(filters=self.down_kernel_num_seq[0],
                                kernel_size=(3, 3),
                                kernel_initializer='he_normal',
                                activation = 'relu',
                                padding='same'),
            keras.layers.BatchNormalization(axis = -1,
                                beta_initializer='zero',
                                gamma_initializer='one'),
            keras.layers.Conv2D(filters=self.num_classes,
                                kernel_size=(3, 3),
                                kernel_initializer='he_normal',
                                activation = 'relu',
                                padding='same'),
            keras.layers.BatchNormalization(axis = -1,
                                beta_initializer='zero',
                                gamma_initializer='one'),
        ])
        self.extra_upsample = None
        if self.pre_encoder:
            self.extra_upsample = keras.Sequential([
            keras.layers.Conv2D(filters=32,
                                kernel_size=(3, 3),
                                kernel_initializer='he_normal',
                                activation = 'relu',
                                padding='same'),
            keras.layers.BatchNormalization(axis = -1,
                                beta_initializer='zero',
                                gamma_initializer='one'),
            keras.layers.Conv2D(filters=32,
                                kernel_size=(3, 3),
                                kernel_initializer='he_normal',
                                activation = 'relu',
                                padding='same'),
            keras.layers.BatchNormalization(axis = -1,
                                beta_initializer='zero',
                                gamma_initializer='one'),
            keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            keras.layers.Conv2D(filters=16,
                                kernel_size=(2, 2),
                                kernel_initializer='he_normal',
                                activation = 'relu',
                                padding='same'),
            keras.layers.BatchNormalization(axis = -1,
                                beta_initializer='zero',
                                gamma_initializer='one')
        ])
        self.pooling = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.softmax = keras.layers.Softmax(axis=-1,name = 'predictor')

    def call(self, inputs):
        tf.cast(inputs, dtype=tf.float32)
        x = inputs

        self.conv_list = []
        self.pool_list = []

        if self.pre_encoder:
            self.pool_list = self.pre_encoder_layers(inputs) # len(self.pool_list) = 4
            self.conv_list = self.pool_list.copy()
        else:
            for k in self.down_conv_layers:
                self.conv_list.append(k(x))
                self.pool_list.append(self.pooling(self.conv_list[-1]))
                x = self.pool_list[-1]

        self.up_samp_list = []
        self.corp_list = []

        if self.pre_encoder:
            self.pool_list.append(self.pooling(self.conv_list[-1])) # turn 64*64 to 32*32        
        self.corp_list.append(self.pool_list[-1]) # init
        tmp_conv_list = self.conv_list[::-1].copy()


        for c,u in zip(tmp_conv_list,self.up_conv_layers):
            self.up_samp_list.append(u(self.corp_list[-1]))
            self.corp_list.append(tf.concat([c,self.up_samp_list[-1]],axis = -1))
        
        pre_output = self.corp_list[-1]
        if self.pre_encoder:
            pre_output = self.extra_upsample(pre_output)
        x = self.output_seq(pre_output)
        output = self.softmax(x)
        return output

    def get_config(self):
        config = {
            'min_kernel_num':self.min_kernel_num,
            'num_classes':self.num_classes,
            'depth':self.depth,
            'pre_encoder':self.pre_encoder,
            'layer_names':self.layer_names
        }
        # cancel base config
        return config