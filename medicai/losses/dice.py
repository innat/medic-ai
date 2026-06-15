from keras import ops

from medicai.utils import DescribeMixin

from .base import BASE_COMMON_ARGS, BaseDiceLoss


class SparseDiceLoss(BaseDiceLoss, DescribeMixin):
    def __init__(
        self,
        from_logits,
        num_classes,
        target_class_ids=None,
        ignore_class_ids=None,
        smooth=1e-7,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class_ids=ignore_class_ids,
            smooth=smooth,
            reduction=reduction,
            name=name or "sparse_dice_loss",
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.softmax(y_pred, axis=-1)
        else:
            return y_pred

    def _process_targets(self, y_true):
        if y_true.shape[-1] == 1:
            y_true = ops.squeeze(y_true, axis=-1)

        y_true = ops.one_hot(y_true, num_classes=self.num_classes)
        return y_true


class CategoricalDiceLoss(BaseDiceLoss, DescribeMixin):
    def __init__(
        self,
        from_logits,
        num_classes,
        target_class_ids=None,
        smooth=1e-7,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class_ids=None,
            smooth=smooth,
            reduction=reduction,
            name=name or "categorical_dice_loss",
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.softmax(y_pred, axis=-1)
        else:
            return y_pred


class BinaryDiceLoss(BaseDiceLoss, DescribeMixin):
    def __init__(
        self,
        from_logits,
        num_classes,
        target_class_ids=None,
        ignore_class_ids=None,
        smooth=1e-7,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        if ignore_class_ids is not None and num_classes > 1:
            raise ValueError(
                "`ignore_class_ids` is only supported when `num_classes=1` "
                "(binary or sparse segmentation). One-hot or multi-label cases "
                "with `num_classes > 1` are not supported."
            )
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class_ids=ignore_class_ids,
            smooth=smooth,
            reduction=reduction,
            name=name or "binary_dice_loss",
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.sigmoid(y_pred)
        else:
            return y_pred


CATEGORICAL_LOSS_DOCSTRING = """Dice loss for categorical one-hot encoded segmentation labels.

This loss computes ``1 - Dice`` for categorical segmentation targets that are
already one-hot encoded. When ``from_logits=True``, the predictions are passed
through a softmax activation before the Dice score is computed. This variant
expects one-hot targets. Use this variant for one-hot multi-class labels. It
does not expose ``ignore_class_ids``.

""" + BASE_COMMON_ARGS.format(
    specific_args="",
    example="""    2D example with ``from_logits=False`` and explicit reductions::

        import keras
        from medicai.losses import CategoricalDiceLoss

        y_true = keras.ops.array(
            [
                [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            ],
            dtype="float32",
        )

        y_pred = keras.ops.array(
            [
                [[[0.5, 1.0, 0.5], [0.0, 0.5, 0.5]]], 
                [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], 
            ],
            dtype="float32",
        )

        loss_none = CategoricalDiceLoss(
            from_logits=False, 
            num_classes=3, 
            reduction="none"
        )
        print(loss_none(y_true, y_pred))
        # [[3.3333337e-01 5.9999996e-01 9.9999988e-01]
        # [5.9604645e-08 5.9604645e-08 6.6666669e-01]]

        loss_sum = CategoricalDiceLoss(
            from_logits=False, 
            num_classes=3, 
            reduction="sum"
        )
        print(loss_sum(y_true, y_pred)) # 2.6

        loss_mean = CategoricalDiceLoss(
            from_logits=False, 
            num_classes=3, 
            reduction="mean"
        )
        print(loss_mean(y_true, y_pred)) # 0.4333333


    3D example with ``from_logits=True``::

        import keras
        from medicai.losses import CategoricalDiceLoss

        y_true = keras.ops.array(
            [
                [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]],
                [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]],
            ],
            dtype="float32",
        )

        y_pred = keras.ops.array(
            [
                [[[[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]]]],
                [[[[15.0, -15.0, -15.0], [-15.0, 15.0, -15.0]]]],
            ],
            dtype="float32",
        )

        loss_mean = CategoricalDiceLoss(
            from_logits=True,
            num_classes=3,
            reduction="mean",
        )

        print(loss_mean(y_true, y_pred)) # 0.36867929

    Example with ``target_class_ids``::

        import keras
        from medicai.losses import CategoricalDiceLoss

        class_indices = keras.ops.array(
            [
                [[5, 0]], 
                [[5, 5]],
            ],
            dtype="int32",
        )
        y_true = keras.ops.one_hot(class_indices, num_classes=6)

        y_pred = keras.ops.array(
            [
                [[[0.0, 0.0, 0.0, 0.0, 0.5, 0.5], [0.5, 0.0, 0.0, 0.0, 0.5, 0.0]]], 
                [[[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]], 
            ],
            dtype="float32",
        )

        loss_target_class = CategoricalDiceLoss(
            from_logits=False,
            num_classes=6,
            target_class_ids=[5],
            reduction="mean",
        )
        print(loss_target_class(y_true, y_pred)) # 0.16666672

    .. note::

        ``CategoricalDiceLoss`` does not expose ``ignore_class_ids`` because the
        targets are already one-hot encoded. To ignore sparse label
        IDs before one-hot conversion, use ``SparseDiceLoss``.""",
    raises="""    ValueError: If ``target_class_ids`` is provided with an unsupported
        type or contains invalid class IDs.""",
    default_name="categorical_dice_loss",
)

SPARSE_LOSS_DOCSTRING = """Dice loss for sparse categorical segmentation labels.

This loss adapts Dice loss to sparse class-index targets by one-hot encoding
them internally. When ``from_logits=True``, the predictions are passed through
softmax before the Dice score is computed.

""" + BASE_COMMON_ARGS.format(
    specific_args="",
    example="""    2D example with ``from_logits=False`` and explicit reductions::

        import keras
        from medicai.losses import SparseDiceLoss

        y_true = keras.ops.array(
            [
                [[[1], [2]]],  
                [[[3], [4]]],
            ],
            dtype="int32",
        )

        y_pred = keras.ops.array(
            [
                [[[0.0, 0.5, 0.0, 0.5, 0.0], [0.0, 0.0, 0.5, 0.0, 0.5]]],
                [[[0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]]],
            ],
            dtype="float32",
        )

        loss_none = SparseDiceLoss(
            from_logits=False, 
            num_classes=5, 
            reduction="none"
        )
        print(loss_none(y_true, y_pred))
        # [[6.6666669e-01 3.3333337e-01 3.3333337e-01 9.9999982e-01 9.9999982e-01]
        # [6.6666669e-01 6.6666669e-01 6.6666669e-01 5.9604645e-08 5.9604645e-08]]

        loss_sum = SparseDiceLoss(
            from_logits=False, 
            num_classes=5, 
            reduction="sum"
        )
        print(loss_sum(y_true, y_pred)) # 5.333333

        loss_mean = SparseDiceLoss(
            from_logits=False, 
            num_classes=5, 
            reduction="mean"
        )
        print(loss_mean(y_true, y_pred)) # 0.5333333


    3D example with ``from_logits=True``::

        import keras
        from medicai.losses import SparseDiceLoss

        y_true = keras.ops.array(
            [
                [[[[1], [2]]]],
                [[[[3], [4]]]],
            ],
            dtype="int32",
        )
        y_pred = keras.ops.array(
            [
                [[[[0.0, 2.0, 0.0, 1.0, 0.0], [0.0, 0.0, 2.0, 0.0, 1.0]]]],
                [[[[-5.0, -1.0, -1.0, 5.0, -2.0], [-2.0, -3.0, -2.0, -1.0, 1.0]]]],
            ],
            dtype="float32",
        )

        loss_mean = SparseDiceLoss(
            from_logits=True,
            num_classes=5,
            reduction="mean",
        )

        print(loss_mean(y_true, y_pred)) # 0.6792049

    Example with ``target_class_ids``::

        import keras
        from medicai.losses import SparseDiceLoss

        y_true = keras.ops.array(
            [
                [[[1], [2]]], 
                [[[3], [4]]],
            ],
            dtype="int32",
        )

        y_pred = keras.ops.array(
            [
                [[[0.0, 0.5, 0.0, 0.5, 0.0], [0.0, 0.0, 0.5, 0.0, 0.5]]],
                [[[0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]]],
            ],
            dtype="float32",
        )

        loss_target_classes = SparseDiceLoss(
            from_logits=False,
            num_classes=5,
            target_class_ids=[1, 3],
            reduction="none",
        )

        print(loss_target_classes(y_true, y_pred))
        # [[3.3333337e-01 9.9999982e-01]
        # [6.6666669e-01 5.9604645e-08]]

    Example with ``ignore_class_ids``::

        import keras
        from medicai.losses import SparseDiceLoss

        y_true = keras.ops.array([[[[1], [2], [0]]]], dtype="int32")
        y_pred = keras.ops.array(
            [[[[0.0, 1.0, 0.0], [0.2, 0.2, 0.6], [1.0, 0.0, 0.0]]]],
            dtype="float32",
        )

        loss = SparseDiceLoss(
            from_logits=False,
            num_classes=3,
            target_class_ids=[1],
            ignore_class_ids=[2],
            reduction="none",
        )

        print(loss(y_true, y_pred))  # [[0.0]]""",
    raises="""    ValueError: If ``target_class_ids`` is provided with an unsupported
        type or contains invalid class IDs.""",
    default_name="sparse_dice_loss",
)


BINARY_LOSS_DOCSTRING = """Dice loss for binary or multi-label segmentation tasks.

This loss computes ``1 - Dice`` for binary or multi-label targets. When
``from_logits=True``, the predictions are passed through a sigmoid activation
before the Dice score is computed.

""" + BASE_COMMON_ARGS.format(
    specific_args="",
    example="""    2D example with ``from_logits=False`` and explicit reductions::

        import keras
        from medicai.losses import BinaryDiceLoss

        y_true = keras.ops.array(
            [
                [[[1.0], [0.0]]],
                [[[1.0], [0.0]]],
            ],
            dtype="float32",
        )

        y_pred = keras.ops.array(
            [
                [[[1.0], [0.0]]],
                [[[0.5], [0.0]]], 
            ],
            dtype="float32",
        )

        # case: reduction="none"
        loss_none = BinaryDiceLoss(
            from_logits=False,
            num_classes=1,
            reduction="none",
        )

        print(loss_none(y_true, y_pred)) # [[5.9604645e-08] [3.3333337e-01]]

        # case: reduction="sum"
        loss_sum = BinaryDiceLoss(
            from_logits=False,
            num_classes=1,
            reduction="sum",
        )

        print(loss_sum(y_true, y_pred)) # 0.33333343

        # case: reduction="mean
        loss_mean = BinaryDiceLoss(
            from_logits=False,
            num_classes=1,
            reduction="mean",
        )

        print(loss_mean(y_true, y_pred)) # 0.16666672


    3D example with ``from_logits=True``::

        y_true = keras.ops.array(
            [
                [[[[1.0], [1.0], [0.0]]]],
                [[[[1.0], [1.0], [0.0]]]],
            ],
            dtype="float32",
        )

        y_pred = keras.ops.array(
            [
                [[[[15.0], [0.0], [-15.0]]]], 
                [[[[-1.3863], [-0.8473], [-15.0]]]],
            ],
            dtype="float32",
        )

        loss_none = BinaryDiceLoss(
            from_logits=True,
            num_classes=1,
            reduction="mean",
        )

        print(loss_none(y_true, y_pred)) # 0.37142906

    Multi-label example with ``target_class_ids``::

        import keras
        from medicai.losses import BinaryDiceLoss

        y_true = keras.ops.array(
            [
                [[[[1.0, 1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0]]]],
                [[[[1.0, 1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0]]]],
            ],
            dtype="float32",
        )

        y_pred = keras.ops.array(
            [
                [[[[1.0, 0.5, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.5, 1.0]]]],  
                [[[[1.0, 1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0]]]],  
            ],
            dtype="float32",
        )

        loss_multi_class = BinaryDiceLoss(
            from_logits=False,
            num_classes=5,
            target_class_ids=[1, 3],
            reduction="none",
        )

        print(loss_multi_class(y_true, y_pred))
        # [[3.3333337e-01 1.4285719e-01] [5.9604645e-08 5.9604645e-08]]

    .. note::

        ``ignore_class_ids`` is only supported when ``num_classes=1``
        (binary or sparse segmentation). One-hot or multi-label cases
        with ``num_classes > 1`` are not supported.""",
    raises="""    ValueError: If ``target_class_ids`` is provided with an unsupported
        type or contains invalid class IDs.
    ValueError: If ``ignore_class_ids`` is used while ``num_classes > 1``.""",
    default_name="binary_dice_loss",
)

CategoricalDiceLoss.__doc__ = CATEGORICAL_LOSS_DOCSTRING
SparseDiceLoss.__doc__ = SPARSE_LOSS_DOCSTRING
BinaryDiceLoss.__doc__ = BINARY_LOSS_DOCSTRING
