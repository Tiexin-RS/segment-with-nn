import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import MaxPool2D
'''
    y_pred: [B, H, W, C] Variable, class softmax probabilities at each prediction (between 0 and 1)
    y_true: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
'''


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true[:, :, :, 1], dtype=tf.int32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        print("----------------------------")
        print(tf.shape(y_pred))
        print(y_pred.shape)
        y_true = tf.one_hot(y_true, y_pred.shape[3])
        print(tf.shape(y_true))
        print(y_true.shape)
        
        y_pred += K.epsilon()
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.math.pow(1 - y_pred, self.gamma) * y_true
        fl = ce * weight * self.alpha
        loss = tf.reduce_mean(fl)
        return loss


class LovaszLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(LovaszLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        C1 = y_pred.shape[3]
        y_pred = tf.reshape(y_pred, (-1, C1))
        y_true = tf.reshape(y_true, (-1, ))
        C2 = y_pred.shape[1]
        losses = []
        present = []
        class_to_sum = list(range(C2))
        for c in class_to_sum:
            fg = tf.cast(tf.equal(y_true, c),
                         y_pred.dtype)  # foreground for class c
            present.append(tf.reduce_sum(fg) > 0)
            class_pred = y_pred[:, c]
            errors = tf.abs(fg - class_pred)
            errors_sorted, perm = tf.nn.top_k(
                errors,
                k=tf.shape(errors)[0],
                name="descending_sort_{}".format(c))
            fg_sorted = tf.gather(fg, perm)
            gts = tf.reduce_sum(fg_sorted)
            intersection = gts - tf.cumsum(fg_sorted)
            union = gts + tf.cumsum(1. - fg_sorted)
            grad = 1. - intersection / union
            grad = tf.concat((grad[0:1], grad[1:] - grad[:-1]), 0)
            losses.append(
                tf.tensordot(errors_sorted,
                             tf.stop_gradient(grad),
                             1,
                             name="loss_class_{}".format(c)))
        losses_tensor = tf.stack(losses)
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
        loss = tf.reduce_mean(losses_tensor)
        return loss


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self,
                 loss_type='jaccard',
                 axis=(1, 2, 3),
                 smooth=1e-5,
                 **kwargs):
        super(DiceLoss, self).__init__(**kwargs)
        self.loss_type = loss_type
        self.axis = axis
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.one_hot(y_true, y_pred.shape[3])
        inse = tf.reduce_sum(y_pred * y_true, axis=self.axis)
        if self.loss_type == 'jaccard':
            l = tf.reduce_sum(y_pred * y_pred, axis=self.axis)
            r = tf.reduce_sum(y_true * y_true, axis=self.axis)
        elif self.loss_type == 'sorensen':
            l = tf.reduce_sum(y_pred, axis=self.axis)
            r = tf.reduce_sum(y_true, axis=self.axis)
        dice = (2. * inse + self.smooth) / (l + r + self.smooth)
        dice = tf.reduce_mean(dice)
        return 1 - dice


class BoundaryLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(BoundaryLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        y_true = tf.one_hot(y_true, y_pred.shape[3])
        y_pred_bd = MaxPool2D(pool_size=(3, 3), strides=(1, 1),
                              padding='same')(1 - y_pred)
        y_true_bd = MaxPool2D(pool_size=(3, 3), strides=(1, 1),
                              padding='same')(1 - y_true)
        y_pred_bd = y_pred_bd - (1 - y_pred)
        y_true_bd = y_true_bd - (1 - y_true)

        y_pred_bd_ext = MaxPool2D(pool_size=(5, 5),
                                  strides=(1, 1),
                                  padding='same')(1 - y_pred)
        y_true_bd_ext = MaxPool2D(pool_size=(5, 5),
                                  strides=(1, 1),
                                  padding='same')(1 - y_true)
        y_pred_bd_ext = y_pred_bd_ext - (1 - y_pred)
        y_true_bd_ext = y_true_bd_ext - (1 - y_true)

        P = K.sum(y_pred_bd * y_true_bd_ext) / K.sum(y_pred_bd) + 1e-7
        R = K.sum(y_true_bd * y_pred_bd_ext) / K.sum(y_true_bd) + 1e-7
        F1_Score = 2 * P * R / (P + R + 1e-7)
        loss = K.mean(1 - F1_Score)
        return loss
