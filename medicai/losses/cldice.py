from keras import ops

from .base import BASE_COMMON_ARGS, BaseCenterlineDiceLoss


class BinaryCenterlineDiceLoss(BaseCenterlineDiceLoss):
    def __init__(
        self,
        from_logits,
        num_classes,
        iters=50,
        target_class_ids=None,
        ignore_class_ids=None,
        memory_efficient_skeleton=True,
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
            iters=iters,
            memory_efficient_skeleton=memory_efficient_skeleton,
            smooth=smooth,
            reduction=reduction,
            name=name or "binary_cldice",
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.sigmoid(y_pred)
        else:
            return y_pred


class SparseCenterlineDiceLoss(BaseCenterlineDiceLoss):
    def __init__(
        self,
        from_logits,
        num_classes,
        iters=50,
        target_class_ids=None,
        ignore_class_ids=None,
        memory_efficient_skeleton=True,
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
            iters=iters,
            memory_efficient_skeleton=memory_efficient_skeleton,
            smooth=smooth,
            reduction=reduction,
            name=name or "sparse_cldice",
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

        y_true = ops.cast(y_true, "int32")
        return ops.one_hot(y_true, self.num_classes, dtype="float32")


class CategoricalCenterlineDiceLoss(BaseCenterlineDiceLoss):
    def __init__(
        self,
        from_logits,
        num_classes,
        iters=50,
        target_class_ids=None,
        memory_efficient_skeleton=True,
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
            iters=iters,
            memory_efficient_skeleton=memory_efficient_skeleton,
            smooth=smooth,
            reduction=reduction,
            name=name or "categorical_cldice",
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.softmax(y_pred, axis=-1)
        else:
            return y_pred

_CLDICE_SPECIFIC_ARGS = """    iters (int, optional): Number of soft-skeletonization iterations.
        Higher values can produce more complete skeletons but increase
        computational cost. Defaults to ``50``.
    memory_efficient_skeleton (bool, optional): If ``True``, gradients are
        stopped through the predicted skeleton to reduce memory usage during
        training. Defaults to ``True``.
"""

_CLDICE_SHARED_OVERVIEW = """
This class implements the core ``1.0 - clDice`` logic for topology-aware
segmentation. It compares predicted and ground-truth soft skeletons and is
especially useful for thin, elongated, or tubular anatomical structures such as
vessels, centerlines, airways, and fibers.

For each selected class channel, clDice is computed as the harmonic mean of:

- **Topology Precision (T_per)**: how much of the predicted skeleton lies
  inside the ground-truth volume.
- **Topology Sensitivity (T_sens)**: how much of the ground-truth skeleton is
  recovered by the predicted volume.

Skeletons are computed using differentiable soft morphological operations.

.. note::

    - Ground-truth skeletons are always computed with gradients disabled.
    - When ``memory_efficient_skeleton=True``, predicted skeletons are also
      detached from the computation graph.
    - This loss assumes channel-last tensors with shape
      ``(batch, [depth], height, width, channels)``.

.. important::

    - Background is typically encoded as class ``0``. To exclude background,
      explicitly set ``target_class_ids`` to foreground classes.
    - If your dataset contains invalid or ignored labels, you must provide
      ``ignore_class_ids`` so those regions are masked before skeletonization.
    - Ignored labels must be spatially masked even when optimizing only
      selected target classes, otherwise they can affect topology precision and
      sensitivity.

"""

CATEGORICAL_CLDICE_DOCSTRING = """Centerline Dice loss for categorical one-hot encoded segmentation labels.

This loss computes ``1 - clDice`` for categorical segmentation targets that are
already one-hot encoded. When ``from_logits=True``, predictions are passed
through a softmax activation before topology-aware overlap is computed.

This variant expects one-hot targets and does not expose ``ignore_class_ids``.
Use ``SparseCenterlineDiceLoss`` if your labels are stored as class indices.

""" + _CLDICE_SHARED_OVERVIEW + BASE_COMMON_ARGS.format(
    specific_args=_CLDICE_SPECIFIC_ARGS,
    example="""    Example with one-hot encoded labels::

        import keras
        from medicai.losses import CategoricalCenterlineDiceLoss

        y_true = keras.ops.array(
            [[[[1.0, 0.0], [0.0, 1.0]]]],
            dtype="float32",
        )
        y_pred = keras.ops.array(
            [[[[0.9, 0.1], [0.2, 0.8]]]],
            dtype="float32",
        )

        loss = CategoricalCenterlineDiceLoss(
            from_logits=False,
            num_classes=2,
            reduction="mean",
        )

        print(loss(y_true, y_pred))""",
    raises="""    ValueError: If ``target_class_ids`` is provided with an unsupported
        type or contains invalid class IDs.""",
    default_name="categorical_cldice",
)

SPARSE_CLDICE_DOCSTRING = """Centerline Dice loss for sparse categorical segmentation labels.

This loss adapts clDice to sparse class-index targets by one-hot encoding them
internally. When ``from_logits=True``, predictions are passed through softmax
before topology-aware overlap is computed.

This variant is appropriate when the ground-truth tensor stores integer class
IDs instead of one-hot vectors.

""" + _CLDICE_SHARED_OVERVIEW + BASE_COMMON_ARGS.format(
    specific_args=_CLDICE_SPECIFIC_ARGS,
    example="""    Example with sparse labels and ignored regions::

        import keras
        from medicai.losses import SparseCenterlineDiceLoss

        y_true = keras.ops.array([[[[1], [2], [0]]]], dtype="int32")
        y_pred = keras.ops.array(
            [[[[0.1, 0.9, 0.0], [0.2, 0.2, 0.6], [0.8, 0.1, 0.1]]]],
            dtype="float32",
        )

        loss = SparseCenterlineDiceLoss(
            from_logits=False,
            num_classes=3,
            target_class_ids=[1],
            ignore_class_ids=[2],
            reduction="mean",
        )

        print(loss(y_true, y_pred))""",
    raises="""    ValueError: If ``target_class_ids`` is provided with an unsupported
        type or contains invalid class IDs.""",
    default_name="sparse_cldice",
)

BINARY_CLDICE_DOCSTRING = """Centerline Dice loss for binary or multi-label segmentation tasks.

This loss computes ``1 - clDice`` for binary or multi-label targets. When
``from_logits=True``, predictions are passed through a sigmoid activation
before topology-aware overlap is computed.

This variant is especially useful for vessel-like or tubular foreground
structures where preserving topology matters more than plain region overlap.

""" + _CLDICE_SHARED_OVERVIEW + BASE_COMMON_ARGS.format(
    specific_args=_CLDICE_SPECIFIC_ARGS,
    example="""    Example with a binary foreground mask::

        import keras
        from medicai.losses import BinaryCenterlineDiceLoss

        y_true = keras.ops.array(
            [[[[1.0], [0.0]], [[1.0], [0.0]]]],
            dtype="float32",
        )
        y_pred = keras.ops.array(
            [[[[0.9], [0.1]], [[0.7], [0.2]]]],
            dtype="float32",
        )

        loss = BinaryCenterlineDiceLoss(
            from_logits=False,
            num_classes=1,
            reduction="mean",
        )

        print(loss(y_true, y_pred))""",
    raises="""    ValueError: If ``target_class_ids`` is provided with an unsupported
        type or contains invalid class IDs.""",
    default_name="binary_cldice",
)


SparseCenterlineDiceLoss.__doc__ = SPARSE_CLDICE_DOCSTRING
BinaryCenterlineDiceLoss.__doc__ = BINARY_CLDICE_DOCSTRING
CategoricalCenterlineDiceLoss.__doc__ = CATEGORICAL_CLDICE_DOCSTRING
