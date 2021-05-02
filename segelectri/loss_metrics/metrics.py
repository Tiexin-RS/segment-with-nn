import tensorflow as tf
from tensorflow.keras import backend as K
import numpy


class MeanIou(tf.keras.metrics.Metric):

    def __init__(self, num_classes=4, **kwargs):
        super(MeanIou, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.total_cm = self.add_weight('total_confusion_matrix',
                                        shape=(num_classes, num_classes),
                                        initializer=tf.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.argmax(y_true, axis=-1), dtype=tf.int64)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), dtype=tf.int64)

        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        current_cm = tf.math.confusion_matrix(y_true,
                                              y_pred,
                                              self.num_classes,
                                              dtype=self._dtype)
        return self.total_cm.assign_add(current_cm)

    def result(self):

        sum_over_row = tf.cast(tf.reduce_sum(self.total_cm, axis=0),
                               dtype=self._dtype)
        sum_over_col = tf.cast(tf.reduce_sum(self.total_cm, axis=1),
                               dtype=self._dtype)
        true_positives = tf.cast(tf.linalg.tensor_diag_part(self.total_cm),
                                 dtype=self._dtype)

        # sum_over_row + sum_over_col = 2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        num_valid_entries = tf.reduce_sum(
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype))

        iou = tf.math.divide_no_nan(true_positives, denominator)

        return tf.math.divide_no_nan(tf.reduce_sum(iou, name='mean_iou'),
                                     num_valid_entries)

    def reset_states(self):
        K.set_value(self.total_cm,
                    numpy.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(MeanIou, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
