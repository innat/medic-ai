from typing import Callable, Optional, Sequence, Union

from keras import ops

from ..base import InvertibleTransform, KeyedTransform, _pop_last_transform_trace
from ..tensor_bundle import TensorBundle
from ..utils import (
    ensure_spatial_tuple,
    get_input_layout_info,
    get_spatial_rank,
    get_spatial_shape_for_layout,
    resolve_input_layout,
    validate_tensor_matches_layout,
)
from .spatial_crop import SpatialCrop


class CropForeground(KeyedTransform, InvertibleTransform):
    """Crop selected tensors to the foreground region of a source tensor.

    This transform detects foreground in ``source_key`` using ``select_fn``,
    builds a bounding box around that region, and applies the same crop to all
    selected tensors. It supports both 2D channel-last tensors ``(H, W, C)``
    and 3D channel-last tensors ``(D, H, W, C)``.
    This transform is sample-only. It does not support batched inputs.

    Foreground is computed by reducing the source tensor across the channel
    dimension and selecting spatial locations where ``select_fn`` evaluates to
    ``True``. Optional margins can then be added, and the final crop size can
    be expanded to be divisible by a requested factor such as a network stride.

    This transform records crop start and end coordinates in metadata when
    ``start_coord_key`` and ``end_coord_key`` are provided. It is invertible
    in the placement sense: :meth:`inverse` pads the cropped tensor back into
    its original spatial canvas using the recorded crop coordinates and
    original spatial shape. This is meant for postprocessing tasks such as
    placing a cropped prediction back into the original image extent. It does
    not reconstruct the discarded background content outside the crop; those
    regions are zero-padded on inversion.

    Args:
        keys: Keys of tensors to crop once the foreground bounding box has been
            estimated.
        source_key: Key of the tensor used to compute the foreground mask. If
            omitted and ``keys`` contains exactly one item, that key is used as
            the foreground source automatically.
        select_fn: Callable that receives the source tensor and returns a
            boolean-like mask used to define foreground.
        channel_indices: Optional subset of source channels used when
            estimating foreground.
        margin: Extra padding added around the detected bounding box. Can be a
            scalar or a per-dimension sequence.
        allow_smaller: If ``True``, allow the crop to shrink against image
            boundaries when margin expansion would exceed the image extent.
        k_divisible: Expand the crop so each spatial dimension is divisible by
            this value or per-dimension sequence.
        start_coord_key: Metadata key used to store crop start coordinates, or
            ``None`` to skip storing them.
        end_coord_key: Metadata key used to store crop end coordinates, or
            ``None`` to skip storing them.
        input_layout: Channel-last tensor layout. Supported values are
            ``"HWC"`` and ``"DHWC"``. Batched layouts are intentionally
            rejected because foreground detection is defined per sample.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:

        TensorFlow backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import CropForeground

            transform = CropForeground(
                keys=["image", "label"],
                source_key="image",
                margin=4,
                input_layout="HWC",
            )

            image = tf.pad(tf.ones((24, 24, 1)), paddings=[[8, 8], [8, 8], [0, 0]])
            label = tf.cast(image > 0, tf.float32)

            result = transform({"image": image, "label": label})
            cropped_image = result["image"]
            cropped_label = result["label"]

        JAX backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "jax"

            import jax.numpy as jnp
            from medicai.transforms import CropForeground

            transform = CropForeground(
                keys=["image"],
                margin=(2, 4, 4),
                k_divisible=2,
                input_layout="DHWC",
            )

            image = jnp.ones((8, 16, 16, 1), dtype=jnp.float32)
            result = transform({"image": image})
            print(result["image"].shape)

        Torch backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "torch"

            import torch
            from medicai.transforms import CropForeground

            transform = CropForeground(
                keys=["image"], source_key="image", input_layout="HWC"
            )
            image = torch.ones((32, 32, 1))
            result = transform({"image": image})
            print(result["image"].shape)

    Returns:
        ``TensorBundle``: The input bundle with cropped tensors, optional crop
        coordinate metadata, and an invertible trace entry appended.

    Raises:
        ValueError: If ``source_key`` is omitted while ``keys`` contains more
            than one entry.
        KeyError: If ``source_key`` or a requested crop key is missing and
            ``allow_missing_keys=False``.
    """

    def __init__(
        self,
        keys: Sequence[str] = ("image", "label"),
        source_key: Optional[str] = None,
        select_fn: Callable = lambda x: x > 0,
        channel_indices: Optional[Sequence[int]] = None,
        margin: Union[Sequence[int], int] = 0,
        allow_smaller: bool = True,
        k_divisible: Union[Sequence[int], int] = 1,
        start_coord_key: Optional[str] = "foreground_start_coord",
        end_coord_key: Optional[str] = "foreground_end_coord",
        *,
        input_layout: str,
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        if source_key is None:
            if len(self.keys) == 1:
                source_key = self.keys[0]
            else:
                raise ValueError(
                    "`source_key` must be provided when `keys` contains more than one item."
                )
        self.source_key = source_key
        self.select_fn = select_fn
        self.channel_indices = channel_indices
        self.margin = margin
        self.allow_smaller = allow_smaller
        self.k_divisible = k_divisible
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            allowed_layouts=("HWC", "DHWC"),
            transform_name=type(self).__name__,
        )
        layout_info = get_input_layout_info(self.input_layout)
        if layout_info.batched:
            raise ValueError(
                f"{type(self).__name__} supports only sample layouts 'HWC' and 'DHWC'."
            )

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        if self.source_key not in bundle.data:
            if self.allow_missing_keys:
                return bundle
            raise KeyError(f"Key '{self.source_key}' not found in input data.")

        source_data = bundle.data[self.source_key]
        validate_tensor_matches_layout(
            source_data,
            self.input_layout,
            transform_name=type(self).__name__,
        )
        spatial_rank = get_spatial_rank(source_data)
        image_shape = get_spatial_shape_for_layout(
            source_data,
            input_layout=self.input_layout,
        )

        if self.channel_indices is not None:
            source_data = ops.take(source_data, self.channel_indices, axis=-1)

        min_coords, max_coords = self.find_bounding_box(source_data, self.select_fn, spatial_rank)
        min_coords, max_coords = self.add_margin(
            min_coords,
            max_coords,
            self.margin,
            image_shape,
            self.allow_smaller,
            spatial_rank,
        )
        min_coords, max_coords = self.make_divisible(
            min_coords,
            max_coords,
            self.k_divisible,
            image_shape,
            spatial_rank,
        )

        crop_size = max_coords - min_coords
        original_shapes = {}
        crop = SpatialCrop(
            keys=self.keys,
            crop_size=1,
            input_layout=self.input_layout,
            allow_missing_keys=self.allow_missing_keys,
        )

        def apply_crop(tensor: object, key: str) -> object:
            original_shapes[key] = get_spatial_shape_for_layout(
                tensor,
                input_layout=self.input_layout,
            )
            # The helper is configured with a placeholder crop size. Use the
            # runtime foreground size rather than that placeholder's static
            # output-shape optimization.
            return crop.crop_tensor(tensor, min_coords, crop_size, static_size=False)

        present_keys = crop.apply_to_present_keys(bundle, apply_crop)

        if self.start_coord_key is not None:
            bundle.meta[self.start_coord_key] = min_coords
        if self.end_coord_key is not None:
            bundle.meta[self.end_coord_key] = max_coords

        self.record_transform(
            bundle,
            {
                "keys": list(present_keys),
                "crop_start": min_coords,
                "crop_size": crop_size,
                "original_shapes": original_shapes,
                "source_key": self.source_key,
                "input_layout": self.input_layout,
            },
        )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = self._get_last_crop_foreground_trace(bundle)
        if trace is None:
            return bundle

        crop_start = trace["params"].get("crop_start")
        original_shapes = trace["params"].get("original_shapes", {})
        crop = SpatialCrop(
            keys=self.keys,
            crop_size=1,
            input_layout=self.input_layout,
            allow_missing_keys=self.allow_missing_keys,
        )

        def apply_inverse_crop(tensor: object, key: str) -> object:
            original_shape = original_shapes.get(key)
            if original_shape is None:
                return tensor
            return crop.pad_to_original_shape(tensor, crop_start, original_shape)

        self.apply_to_present_keys(
            bundle,
            apply_inverse_crop,
            keys=trace["params"].get("keys", []),
        )
        return bundle

    def find_bounding_box(
        self,
        image: object,
        select_fn: Callable,
        spatial_rank: int,
    ) -> tuple[object, object]:
        """Find the bounding box of the foreground in the image."""
        mask = ops.any(select_fn(image), axis=-1)
        has_foreground = ops.any(mask)
        spatial_shape = ops.cast(
            get_spatial_shape_for_layout(image, input_layout=self.input_layout),
            "int32",
        )

        def empty_bbox():
            return (
                ops.zeros((spatial_rank,), dtype="int32"),
                spatial_shape,
            )

        def foreground_bbox():
            min_coords = []
            max_coords = []
            for axis in range(spatial_rank):
                other_axes = tuple(i for i in range(spatial_rank) if i != axis)
                axis_presence = mask
                # Reduce one axis at a time in descending order. Some Keras
                # backends do not support a tuple of reduction axes, and the
                # descending order keeps the target axis index stable.
                for reduce_axis in reversed(other_axes):
                    axis_presence = ops.any(axis_presence, axis=reduce_axis)
                axis_presence_i32 = ops.cast(axis_presence, "int32")
                axis_size = ops.shape(axis_presence_i32)[0]
                start = ops.argmax(axis_presence_i32, axis=0)
                end = axis_size - ops.argmax(ops.flip(axis_presence_i32, axis=0), axis=0)
                min_coords.append(ops.cast(start, "int32"))
                max_coords.append(ops.cast(end, "int32"))
            min_coords = ops.stack(min_coords, axis=0)
            max_coords = ops.stack(max_coords, axis=0)
            return min_coords, max_coords

        return ops.cond(has_foreground, foreground_bbox, empty_bbox)

    def add_margin(
        self,
        min_coords: object,
        max_coords: object,
        margin: Union[Sequence[int], int],
        image_shape: object,
        allow_smaller: bool,
        spatial_rank: int,
    ) -> tuple[object, object]:
        """Add margin to the bounding box while staying inside image bounds."""
        margin = ops.convert_to_tensor(
            ensure_spatial_tuple(margin, spatial_rank, "margin"),
            dtype="int32",
        )

        image_shape = ops.cast(image_shape, "int32")
        requested_min = ops.cast(min_coords, "int32") - margin
        requested_max = ops.cast(max_coords, "int32") + margin
        min_coords = ops.maximum(requested_min, 0)
        max_coords = ops.minimum(requested_max, image_shape)

        if not allow_smaller:
            requested_size = requested_max - requested_min
            current_size = max_coords - min_coords
            deficit = ops.maximum(requested_size - current_size, 0)

            shift_left = ops.minimum(min_coords, deficit)
            min_coords = min_coords - shift_left
            deficit = deficit - shift_left

            shift_right = ops.minimum(image_shape - max_coords, deficit)
            max_coords = max_coords + shift_right

        return min_coords, max_coords

    def make_divisible(
        self,
        min_coords: object,
        max_coords: object,
        k_divisible: Union[Sequence[int], int],
        image_shape: object,
        spatial_rank: int,
    ) -> tuple[object, object]:
        """Expand the bounding box so its size is divisible by ``k_divisible``."""
        k_divisible = ops.convert_to_tensor(
            ensure_spatial_tuple(k_divisible, spatial_rank, "k_divisible"),
            dtype="int32",
        )

        size = max_coords - min_coords
        remainder = size % k_divisible
        padding = ops.where(remainder != 0, k_divisible - remainder, 0)
        image_shape = ops.cast(image_shape, "int32")
        requested_max = max_coords + padding
        max_coords = ops.minimum(requested_max, image_shape)
        overflow = ops.maximum(requested_max - max_coords, 0)
        min_coords = ops.maximum(min_coords - overflow, 0)
        return min_coords, max_coords

    def _get_last_crop_foreground_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)
