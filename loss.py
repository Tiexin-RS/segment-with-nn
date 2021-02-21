def dice_loss(y_pred, y_true):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    return 1 - (numerator + 1)/ (denominator + 1)
    
def dice_coef(y_true,y_pred):
    sum1 = 2*tf.reduce_sum(y_true*y_pred,axis=(0,1,2))
    sum2 = tf.reduce_sum(y_true**2+y_pred**2,axis=(0,1,2))
    dice = (sum1+0.1)/(sum2+0.1)
    dice = tf.reduce_mean(dice)
    return dice
def dice_coef_loss(y_true,y_pred):
    return 1.-dice_coef(y_true,y_pred)

# https://github.com/rishavroy1264bitmesra/Tensorflow-Boundary-Loss-for-Remote-Sensing-Imagery-Semantic-Segmentation/blob/master/boundary_loss.py
 
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
 
#Shape of semantic segmentation mask
OUTPUT_SHAPE = (608, 608, 1)
 
def segmentation_boundary_loss(y_true, y_pred):
    """
    Paper Implemented : https://arxiv.org/abs/1905.07852
    Using Binary Segmentation mask, generates boundary mask on fly and claculates boundary loss.
    :param y_true:
    :param y_pred:
    :return:
    """
    y_pred_bd = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same', input_shape=OUTPUT_SHAPE)(1 - y_pred)
    y_true_bd = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same', input_shape=OUTPUT_SHAPE)(1 - y_true)
    y_pred_bd = y_pred_bd - (1 - y_pred)
    y_true_bd = y_true_bd - (1 - y_true)
 
    y_pred_bd_ext = layers.MaxPooling2D((5, 5), strides=(1, 1), padding='same', input_shape=OUTPUT_SHAPE)(1 - y_pred)
    y_true_bd_ext = layers.MaxPooling2D((5, 5), strides=(1, 1), padding='same', input_shape=OUTPUT_SHAPE)(1 - y_true)
    y_pred_bd_ext = y_pred_bd_ext - (1 - y_pred)
    y_true_bd_ext = y_true_bd_ext - (1 - y_true)
 
    P = K.sum(y_pred_bd * y_true_bd_ext) / K.sum(y_pred_bd) + 1e-7
    R = K.sum(y_true_bd * y_pred_bd_ext) / K.sum(y_true_bd) + 1e-7
    F1_Score = 2 * P * R / (P + R + 1e-7)
    # print(f'Precission: {P.eval()}, Recall: {R.eval()}, F1: {F1_Score.eval()}')
    loss = K.mean(1 - F1_Score)
    # print(f"Loss:{loss.eval()}")
    return loss

# 注意，alpha是一个和你的分类类别数量相等的向量；
alpha=[[1], [1], [1], [1]]

def focal_loss(logits, labels, alpha，epsilon = 1.e-7,
                   gamma=2.0, 
                   multi_dim = False):
        '''
        :param logits:  [batch_size, n_class]
        :param labels: [batch_size]  not one-hot !!!
        :return: -alpha*(1-y)^r * log(y)
        它是在哪实现 1- y 的？ 通过gather选择的就是1-p,而不是通过计算实现的；
        logits soft max之后是多个类别的概率，也就是二分类时候的1-P和P；多分类的时候不是1-p了；

        怎么把alpha的权重加上去？
        通过gather把alpha选择后变成batch长度，同时达到了选择和维度变换的目的

        是否需要对logits转换后的概率值进行限制？
        需要的，避免极端情况的影响

        针对输入是 (N，P，C )和  (N，P)怎么处理？
        先把他转换为和常规的一样形状，（N*P，C） 和 （N*P,）

        bug:
        ValueError: Cannot convert an unknown Dimension to a Tensor: ?
        因为输入的尺寸有时是未知的，导致了该bug,如果batchsize是确定的，可以直接修改为batchsize

        '''


        if multi_dim:
            logits = tf.reshape(logits, [-1, logits.shape[2]])
            labels = tf.reshape(labels, [-1])

        # (Class ,1)
        alpha = tf.constant(alpha, dtype=tf.float32)

        labels = tf.cast(labels, dtype=tf.int32)
        logits = tf.cast(logits, tf.float32)
        # (N,Class) > N*Class
        softmax = tf.reshape(tf.nn.softmax(logits), [-1])  # [batch_size * n_class]
        # (N,) > (N,) ,但是数值变换了，变成了每个label在N*Class中的位置
        labels_shift = tf.range(0, logits.shape[0]) * logits.shape[1] + labels
        #labels_shift = tf.range(0, batch_size*32) * logits.shape[1] + labels
        # (N*Class,) > (N,)
        prob = tf.gather(softmax, labels_shift)
        # 预防预测概率值为0的情况  ; (N,)
        prob = tf.clip_by_value(prob, epsilon, 1. - epsilon)
        # (Class ,1) > (N,)
        alpha_choice = tf.gather(alpha, labels)
        # (N,) > (N,)
        weight = tf.pow(tf.subtract(1., prob), gamma)
        weight = tf.multiply(alpha_choice, weight)
        # (N,) > 1
        loss = -tf.reduce_mean(tf.multiply(weight, tf.log(prob)))
        return loss
    
def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    if per_image:
        def treat_image(prob_lab):
            prob, lab = prob_lab
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = flatten_probas(prob, lab, ignore, order)
            return lovasz_softmax_flat(prob, lab, classes=classes)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore, order), classes=classes)
    return loss
