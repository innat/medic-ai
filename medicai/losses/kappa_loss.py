from typing import Optional, Union
from typeguard import typechecked

import numpy as np
import tensorflow as tf
from tensorflow import keras

Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]


# ref. This CohanKappa layer is ported from TensorFlow Addons
class WeightedKappaLoss(keras.losses.Loss):
    @typechecked
    def __init__(
        self,
        num_classes: int,
        weightage: Optional[str] = "quadratic",
        name: Optional[str] = "cohen_kappa_loss",
        epsilon: Optional[Number] = 1e-6,
        reduction: str = keras.losses.Reduction.NONE,
    ):
        super().__init__(name=name, reduction=reduction)

        if weightage not in ("linear", "quadratic"):
            raise ValueError("Unknown kappa weighting type.")

        self.weightage = weightage
        self.num_classes = num_classes
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        label_vec = tf.range(num_classes, dtype=tf.keras.backend.floatx())
        self.row_label_vec = tf.reshape(label_vec, [1, num_classes])
        self.col_label_vec = tf.reshape(label_vec, [num_classes, 1])
        col_mat = tf.tile(self.col_label_vec, [1, num_classes])
        row_mat = tf.tile(self.row_label_vec, [num_classes, 1])
        if weightage == "linear":
            self.weight_mat = tf.abs(col_mat - row_mat)
        else:
            self.weight_mat = (col_mat - row_mat) ** 2

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=self.col_label_vec.dtype)
        y_pred = tf.cast(y_pred, dtype=self.weight_mat.dtype)
        batch_size = tf.shape(y_true)[0]
        cat_labels = tf.matmul(y_true, self.col_label_vec)
        cat_label_mat = tf.tile(cat_labels, [1, self.num_classes])
        row_label_mat = tf.tile(self.row_label_vec, [batch_size, 1])
        if self.weightage == "linear":
            weight = tf.abs(cat_label_mat - row_label_mat)
        else:
            weight = (cat_label_mat - row_label_mat) ** 2
        numerator = tf.reduce_sum(weight * y_pred)
        label_dist = tf.reduce_sum(y_true, axis=0, keepdims=True)
        pred_dist = tf.reduce_sum(y_pred, axis=0, keepdims=True)
        w_pred_dist = tf.matmul(self.weight_mat, pred_dist, transpose_b=True)
        denominator = tf.reduce_sum(tf.matmul(label_dist, w_pred_dist))
        denominator /= tf.cast(batch_size, dtype=denominator.dtype)
        loss = tf.math.divide_no_nan(numerator, denominator)
        return tf.math.log(loss + self.epsilon)

    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "weightage": self.weightage,
            "epsilon": self.epsilon,
        }
        base_config = super().get_config()
        return {**base_config, **config}
