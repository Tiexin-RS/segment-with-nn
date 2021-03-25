import tensorflow as tf
from tensorflow import keras


class downsamp_conv(keras.layers.Layer):
    def __init__(self, filters_num=128, **kwargs):
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

    # def get_config(self):


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

    # def get_config(self):


class Unet(keras.layers.Layer):
    def __init__(self, min_kernel_num=64, num_classes=10, **kwargs):
        """Initialize Unet
        Args:
            min_kernel_num:num of filters in the toppest layer
        """
        super(Unet, self).__init__(**kwargs)
        self.min_kernel_num = min_kernel_num // 1  #assure that is a int
        self.kernel_num_seq = [
            self.min_kernel_num, self.min_kernel_num * 2,
            self.min_kernel_num * 4, self.min_kernel_num * 8,
            self.min_kernel_num * 16
        ]
        self.num_classes = num_classes

    def build(self, input_shape):
        self.shape = input_shape

        # use loop to wrap a list
        self.down_conv_layers = []
        for i in range(4):
            self.down_conv_layers.append(downsamp_conv(self.kernel_num_seq[i]))

        # use loop to wrap a list
        self.up_conv_layers = []
        for i in range(4):
            self.up_conv_layers.append(upsamp_conv(self.kernel_num_seq[-i -
                                                                       1]))

        self.output_seq = keras.Sequential([
            keras.layers.Conv2D(filters=self.kernel_num_seq[0],
                                kernel_size=(3, 3),
                                padding='same'),
            keras.layers.Conv2D(filters=self.kernel_num_seq[0],
                                kernel_size=(3, 3),
                                padding='same'),
            keras.layers.Conv2D(filters=self.num_classes,
                                kernel_size=(3, 3),
                                padding='same')
        ])

        self.pooling = keras.layers.MaxPooling2D(pool_size=(2, 2))

    def call(self, inputs):
        tf.cast(inputs, dtype=tf.float32)
        # print(inputs)
        x = inputs

        self.conv_list = []
        self.pool_list = []

        for i in range(4):
            self.conv_list.append(self.down_conv_layers[i](x))
            self.pool_list.append(self.pooling(self.conv_list[-1]))
            x = self.pool_list[-1]

        self.up_samp_list = []
        self.corp_list = []
        self.up_samp_list.append(self.up_conv_layers[0](
            self.pool_list[-1]))  # init

        for i in range(3):
            self.corp_list.append(
                tf.concat([self.conv_list[-i - 1], self.up_samp_list[i]],
                          axis=-1))
            self.up_samp_list.append(self.up_conv_layers[i + 1](
                self.corp_list[i]))

        self.corp_list.append(
            tf.concat([self.conv_list[0], self.up_samp_list[-1]], axis=-1))
        output = self.output_seq(self.corp_list[-1])
        return output
