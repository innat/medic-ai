from typing import Any, Sequence

from keras import ops

from ..base import InvertibleTransform, KeyedTransform, _pop_last_transform_trace
from ..tensor_bundle import TensorBundle
from ..utils import (
    ensure_spatial_tuple,
    get_input_layout_info,
    get_spatial_shape_for_layout,
    resolve_input_layout,
    validate_tensor_matches_layout,
)


class SpatialCrop(KeyedTransform, InvertibleTransform):
    """Deterministically crop a spatial region from selected tensors.

    This transform crops channel-last tensors using either a crop
    ``crop_start`` or a crop ``crop_center`` together with ``crop_size``.
    Depending on ``input_layout``, it supports:

    - sample 2D tensors shaped ``(H, W, C)``
    - sample 3D tensors shaped ``(D, H, W, C)``
    - batch 2D tensors shaped ``(B, H, W, C)``
    - batch 3D tensors shaped ``(B, D, H, W, C)``

    ``crop_size`` is interpreted per spatial dimension. Non-positive values are
    treated as "use the full available size" for that dimension after bounds
    checking. Crop bounds are clipped so the output always stays within the
    selected tensor's spatial extent. In batch mode, one shared deterministic
    crop is applied across the full batch.

    Args:
        keys: Keys of the tensors to crop.
        crop_size: Desired spatial crop size.
        crop_start: Spatial start coordinates of the crop. Mutually exclusive
            with ``crop_center``.
        crop_center: Spatial center coordinates of the crop. Mutually exclusive
            with ``crop_start``.
        input_layout: Channel-last tensor layout. Supported values are
            ``"HWC"``, ``"DHWC"``, ``"BHWC"``, and ``"BDHWC"``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Keras selects its backend before the first Keras import. Each example
        below is an independent process.

        TensorFlow backend:

        .. code-block:: python

            import os

            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import SpatialCrop

            transform = SpatialCrop(
                keys=["image", "label"],
                crop_size=(32, 64, 64),
                crop_center=(24, 48, 48),
                input_layout="DHWC",
            )

            image = tf.random.normal((48, 96, 96, 1))
            label = tf.random.uniform(
                (48, 96, 96, 1), maxval=2, dtype=tf.int32
            )
            result = transform({"image": image, "label": label})

            print(result["image"].shape)
            print(result["label"].shape)

        JAX backend:

        .. code-block:: python

            import os

            os.environ["KERAS_BACKEND"] = "jax"

            import jax
            from medicai.transforms import SpatialCrop

            transform = SpatialCrop(
                keys=["image", "label"],
                crop_size=(64, 64),
                crop_center=(48, 48),
                input_layout="HWC",
            )

            image = jax.random.normal(
                jax.random.PRNGKey(7), shape=(96, 96, 1)
            )
            result = transform({"image": image})
            print(result["image"].shape)

        Torch backend:

        .. code-block:: python

            import os

            os.environ["KERAS_BACKEND"] = "torch"

            import torch
            from medicai.transforms import SpatialCrop

            transform = SpatialCrop(
                keys=["image"],
                crop_size=(32, 32),
                crop_center=(32, 32),
                input_layout="BHWC",
            )
            torch.manual_seed(7)
            batch = torch.randn((2, 64, 64, 1))
            result = transform({"image": batch})
            print(result["image"].shape)

    ``inverse()`` is a placement inverse: it pads the cropped tensor back into
    its original spatial canvas using the recorded crop coordinates. This is
    especially useful for mapping model predictions back to the pre-crop
    layout. It does not reconstruct values that were discarded outside the crop
    region; those locations are filled with zeros.

    Returns:
        ``TensorBundle``: The input bundle with cropped tensors and an
        invertible transform trace entry appended when at least one
        selected key is present.

    Raises:
        ValueError: If both ``crop_start`` and ``crop_center`` are provided, or
            if a spatial parameter does not match the selected tensor rank.
        KeyError: If a requested key is missing and
            ``allow_missing_keys=False``.
    """

    def __init__(
        self,
        keys: Sequence[str],
        crop_size: int | Sequence[int],
        crop_start: Sequence[int] | None = None,
        crop_center: Sequence[int] | None = None,
        *,
        input_layout: str,
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        if crop_start is not None and crop_center is not None:
            raise ValueError("Only one of `crop_start` or `crop_center` may be provided.")

        self.crop_size = crop_size
        self.crop_start = tuple(crop_start) if crop_start is not None else None
        self.crop_center = tuple(crop_center) if crop_center is not None else None
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            transform_name=type(self).__name__,
        )
        self.layout_info = get_input_layout_info(self.input_layout)

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        crop_starts = {}
        crop_sizes = {}
        original_shapes = {}

        def apply_crop(tensor: Any, key: str) -> Any:
            starts, crop_size = self.compute_crop_bounds(tensor)
            crop_starts[key] = starts
            crop_sizes[key] = crop_size
            original_shapes[key] = get_spatial_shape_for_layout(
                tensor,
                input_layout=self.input_layout,
            )
            return self.crop_tensor(tensor, starts, crop_size)

        present_keys = self.apply_to_present_keys(bundle, apply_crop)

        if crop_starts:
            self.record_transform(
                bundle,
                {
                    "keys": list(present_keys),
                    "crop_start": crop_starts,
                    "crop_size": crop_sizes,
                    "original_shapes": original_shapes,
                    "input_layout": self.input_layout,
                },
            )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = self._get_last_spatial_crop_trace(bundle)
        if trace is None:
            return bundle

        crop_starts = trace["params"].get("crop_start", {})
        original_shapes = trace["params"].get("original_shapes", {})

        def apply_inverse_crop(tensor: Any, key: str) -> Any:
            crop_start = crop_starts.get(key)
            original_shape = original_shapes.get(key)
            if crop_start is None or original_shape is None:
                return tensor
            return self.pad_to_original_shape(tensor, crop_start, original_shape)

        self.apply_to_present_keys(
            bundle,
            apply_inverse_crop,
            keys=trace["params"].get("keys", []),
        )
        return bundle

    def compute_crop_bounds(self, tensor: Any) -> tuple[Any, Any]:
        """Compute bounded crop start coordinates and crop size.

        Args:
            tensor: Channel-last 2D or 3D sample tensor to crop.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: A pair ``(starts, crop_size)`` where
            both tensors have one value per spatial dimension.
        """
        layout = validate_tensor_matches_layout(
            tensor,
            self.input_layout,
            transform_name=type(self).__name__,
        )
        spatial_rank = layout.spatial_rank
        spatial_shape = get_spatial_shape_for_layout(
            tensor,
            input_layout=self.input_layout,
        )
        crop_size = ops.convert_to_tensor(
            ensure_spatial_tuple(self.crop_size, spatial_rank, "crop_size"),
            dtype="int32",
        )
        crop_size = ops.where(crop_size > 0, crop_size, spatial_shape)
        crop_size = ops.minimum(crop_size, spatial_shape)

        if self.crop_start is not None:
            starts = ops.convert_to_tensor(
                ensure_spatial_tuple(self.crop_start, spatial_rank, "crop_start"),
                dtype="int32",
            )
        elif self.crop_center is not None:
            center = ops.convert_to_tensor(
                ensure_spatial_tuple(self.crop_center, spatial_rank, "crop_center"),
                dtype="int32",
            )
            starts = ops.maximum(center - crop_size // 2, 0)
        else:
            starts = ops.zeros((spatial_rank,), dtype="int32")

        ends = ops.minimum(starts + crop_size, spatial_shape)
        starts = ops.maximum(ends - crop_size, 0)
        return starts, crop_size

    def crop_tensor(self, tensor: Any, starts: Any, crop_size: Any) -> Any:
        """Crop one tensor using TensorFlow slicing.

        Args:
            tensor: Channel-last 2D or 3D sample tensor.
            starts: Start indices for each spatial dimension.
            crop_size: Crop size for each spatial dimension.

        Returns:
            ``tf.Tensor``: The cropped tensor with the original channel
            dimension preserved.
        """
        if self.layout_info.batched:
            begin = ops.concatenate(
                [
                    ops.convert_to_tensor([0], dtype="int32"),
                    starts,
                    ops.convert_to_tensor([0], dtype="int32"),
                ],
                axis=0,
            )
            size = ops.concatenate(
                [
                    ops.reshape(ops.shape(tensor)[0], (1,)),
                    crop_size,
                    ops.reshape(ops.shape(tensor)[-1], (1,)),
                ],
                axis=0,
            )
        else:
            begin = ops.concatenate(
                [starts, ops.convert_to_tensor([0], dtype="int32")],
                axis=0,
            )
            size = ops.concatenate(
                [crop_size, ops.reshape(ops.shape(tensor)[-1], (1,))],
                axis=0,
            )
        return ops.slice(tensor, start_indices=begin, shape=size)

    def pad_to_original_shape(
        self,
        tensor: Any,
        crop_start: Any,
        original_shape: Any,
    ) -> Any:
        """Pad a cropped tensor back into its original spatial canvas."""
        crop_start = ops.cast(crop_start, "int32")
        original_shape = ops.cast(original_shape, "int32")
        current_shape = get_spatial_shape_for_layout(
            tensor,
            input_layout=self.input_layout,
        )
        pad_before = crop_start
        pad_after = original_shape - crop_start - current_shape
        spatial_paddings = ops.stack([pad_before, pad_after], axis=1)
        if self.layout_info.batched:
            paddings = ops.concatenate(
                [
                    ops.convert_to_tensor([[0, 0]], dtype="int32"),
                    spatial_paddings,
                    ops.convert_to_tensor([[0, 0]], dtype="int32"),
                ],
                axis=0,
            )
        else:
            paddings = ops.concatenate(
                [
                    spatial_paddings,
                    ops.convert_to_tensor([[0, 0]], dtype="int32"),
                ],
                axis=0,
            )
        return ops.pad(tensor, paddings)

    def _get_last_spatial_crop_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)
