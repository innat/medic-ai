"""Implements Cohen's Kappa."""

from typing import Optional, Union

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras import ops
from keras.metrics import Metric
from typeguard import typechecked

FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]
AcceptableDTypes = Union[tf.DType, np.dtype, type, int, str, None]


# ref. This CohanKappa layer is ported from TensorFlow Addons
class CohenKappa(Metric):
    @typechecked
    def __init__(
        self,
        num_classes: FloatTensorLike,
        name: str = "cohen_kappa",
        weightage: Optional[str] = None,
        sparse_labels: bool = False,
        regression: bool = False,
        dtype: AcceptableDTypes = None,
    ):
        """Creates a `CohenKappa` instance."""
        super().__init__(name=name, dtype=dtype)

        if weightage not in (None, "linear", "quadratic"):
            raise ValueError("Unknown kappa weighting type.")

        if num_classes == 2:
            self._update = self._update_binary_class_model
        elif num_classes > 2:
            self._update = self._update_multi_class_model
        else:
            raise ValueError(
                """Number of classes must be
                              greater than or euqal to two"""
            )

        self.weightage = weightage
        self.num_classes = num_classes
        self.regression = regression
        self.sparse_labels = sparse_labels
        self.conf_mtx = self.add_weight(
            "conf_mtx",
            shape=(self.num_classes, self.num_classes),
            initializer=keras.initializers.zeros,
            dtype='float32',
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        return self._update(y_true, y_pred, sample_weight)

    def _update_binary_class_model(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(y_true, dtype='int64')
        y_pred = ops.cast(y_pred, dtype='float32')
        y_pred = ops.cast(y_pred > 0.5, dtype='int64')
        return self._update_confusion_matrix(y_true, y_pred, sample_weight)

    @tf.function
    def _update_multi_class_model(self, y_true, y_pred, sample_weight=None):
        v = ops.argmax(y_true, axis=1) if not self.sparse_labels else y_true
        y_true = ops.cast(v, dtype='int64')

        y_pred = self._cast_ypred(y_pred)

        return self._update_confusion_matrix(y_true, y_pred, sample_weight)

    @tf.function
    def _cast_ypred(self, y_pred):
        if ops.rank(y_pred) > 1:
            if not self.regression:
                y_pred = ops.cast(ops.argmax(y_pred, axis=-1), dtype='int64')
            else:
                y_pred = ops.math.round(ops.math.abs(y_pred))
                y_pred = ops.cast(y_pred, dtype='int64')
        else:
            y_pred = ops.cast(y_pred, dtype='int64')
        return y_pred

    @tf.function
    def _safe_squeeze(self, y):
        y = ops.squeeze(y)

        # Check for scalar result
        if ops.rank(y) == 0:
            y = ops.expand_dims(y, 0)

        return y

    def _update_confusion_matrix(self, y_true, y_pred, sample_weight):
        y_true = self._safe_squeeze(y_true)
        y_pred = self._safe_squeeze(y_pred)

        new_conf_mtx = ops.math.confusion_matrix(
            labels=y_true,
            predictions=y_pred,
            num_classes=self.num_classes,
            weights=sample_weight,
            dtype='float32',
        )

        return self.conf_mtx.assign_add(new_conf_mtx)

    def result(self):
        nb_ratings = ops.shape(self.conf_mtx)[0]
        weight_mtx = ops.ones([nb_ratings, nb_ratings], dtype=tf.float32)

        # 2. Create a weight matrix
        if self.weightage is None:
            diagonal = ops.zeros([nb_ratings], dtype='float32')
            weight_mtx = ops.linalg.set_diag(weight_mtx, diagonal=diagonal)
        else:
            weight_mtx += ops.cast(ops.range(nb_ratings), dtype='float32')
            weight_mtx = ops.cast(weight_mtx, dtype=self.dtype)

            if self.weightage == "linear":
                weight_mtx = ops.abs(weight_mtx - ops.transpose(weight_mtx))
            else:
                weight_mtx = ops.pow((weight_mtx - ops.transpose(weight_mtx)), 2)

        weight_mtx = ops.cast(weight_mtx, dtype=self.dtype)

        # 3. Get counts
        actual_ratings_hist = ops.sum(self.conf_mtx, axis=1)
        pred_ratings_hist = ops.sum(self.conf_mtx, axis=0)

        # 4. Get the outer product
        out_prod = pred_ratings_hist[..., None] * actual_ratings_hist[None, ...]

        # 5. Normalize the confusion matrix and outer product
        conf_mtx = self.conf_mtx / ops.sum(self.conf_mtx)
        out_prod = out_prod / ops.sum(out_prod)

        conf_mtx = ops.cast(conf_mtx, dtype=self.dtype)
        out_prod = ops.cast(out_prod, dtype=self.dtype)

        # 6. Calculate Kappa score
        numerator = ops.sum(conf_mtx * weight_mtx)
        denominator = ops.sum(out_prod * weight_mtx)
        return ops.cond(
            ops.math.is_nan(denominator),
            true_fn=lambda: 0.0,
            false_fn=lambda: 1 - (numerator / denominator),
        )

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "weightage": self.weightage,
            "sparse_labels": self.sparse_labels,
            "regression": self.regression,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def reset_state(self):
        """Resets all of the metric state variables."""

        for v in self.variables:
            K.set_value(
                v,
                np.zeros((self.num_classes, self.num_classes), v.dtype.as_numpy_dtype),
            )

    def reset_states(self):
        # Backwards compatibility alias of `reset_state`. New classes should
        # only implement `reset_state`.
        # Required in Tensorflow < 2.5.0
        return self.reset_state()
