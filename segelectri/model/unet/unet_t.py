import tensorflow as tf
from tensorflow import keras


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
                                padding='same'),
            keras.layers.Conv2D(filters=self.filters_num,
                                kernel_size=(3, 3),
                                padding='same')
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
                                padding='same'),
            keras.layers.Conv2D(filters=self.filters_num,
                                kernel_size=(3, 3),
                                padding='same'),
            keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            keras.layers.Conv2D(filters=self.filters_num / 2,
                                kernel_size=(2, 2),
                                padding='same')
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


class Unet(keras.layers.Layer):
    def __init__(self, min_kernel_num=64, num_classes=10, **kwargs):
        """Initialize Unet
        Args:
            min_kernel_num:num of filters in the toppest layer
        """
        super(Unet, self).__init__(**kwargs)
        self.min_kernel_num = min_kernel_num // 1  #assure that is a int
        self.num_classes = num_classes
        self.depth = depth

        self.down_kernel_num_seq = [] # default 4 layers
        #根据层数生成
        for i in range(depth):
            self.down_kernel_num_seq.append(min_kernel_num*(2**i))
        self.up_kernel_num_seq = self.down_kernel_num_seq.copy() 
        self.up_kernel_num_seq.append(self.down_kernel_num_seq[-1]*2)
        self.up_kernel_num_seq = self.up_kernel_num_seq[:0:-1] # 反序list 
        # 并且减去一层最上层 交给output处理以进行语义分割

    def build(self, input_shape):
        self.shape = input_shape

        # use loop to wrap a list
        self.down_conv_layers = []
        for k in self.down_kernel_num_seq:
            self.down_conv_layers.append(downsamp_conv(k))
        
        # use loop to wrap a list
        self.up_conv_layers = []
        
        for k in self.up_kernel_num_seq:
            self.up_conv_layers.append(upsamp_conv(k))

        self.output_seq = keras.Sequential([
            keras.layers.Conv2D(filters=self.down_kernel_num_seq[0],
                                kernel_size=(3, 3),
                                padding='same'),
            keras.layers.Conv2D(filters=self.down_kernel_num_seq[0],
                                kernel_size=(3, 3),
                                padding='same'),
            keras.layers.Conv2D(filters=self.num_classes,
                                kernel_size=(3, 3),
                                padding='same')
        ])

        self.pooling = keras.layers.MaxPooling2D(pool_size=(2, 2))

    def call(self, inputs):
        tf.cast(inputs, dtype=tf.float32)
        x = inputs

        self.conv_list = []
        self.pool_list = []

        for k in self.down_conv_layers:
            self.conv_list.append(k(x))
            self.pool_list.append(self.pooling(self.conv_list[-1]))
            x = self.pool_list[-1]

        self.up_samp_list = []
        self.corp_list = []
        
        self.corp_list.append(self.pool_list[-1]) # init
        tmp_conv_list = self.conv_list[::-1]

        for c,u in zip(tmp_conv_list,self.up_conv_layers):
            self.up_samp_list.append(u(self.corp_list[-1]))
            self.corp_list.append(tf.concat([c,self.up_samp_list[-1]],axis = -1))
        
        output = self.output_seq(self.corp_list[-1])
        return output

    def get_config(self):
        config = {
            'min_kernel_num':self.min_kernel_num,
            'num_classes':self.num_classes,
            'depth':self.depth
        }
        base_config = super(Unet,self).get_config()
        return dict(list(base_config.items())+list(config.items()))
