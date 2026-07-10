from typing import Callable, Optional, Sequence, Union

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform, _pop_last_transform_trace
from ..tensor_bundle import TensorBundle
from ..utils import (
    ensure_spatial_tuple,
    get_spatial_rank,
    get_spatial_shape,
    largest_component_mask,
)
from .spatial_crop import SpatialCrop


def largest_component_mask_tf(mask: tf.Tensor) -> tf.Tensor:
    """tf.py_function bridge so the pure-numpy algorithm can run inside
    CropForeground's tf.cond/tf.where graph-mode bounding-box path."""

    def _np_fn(mask_t):
        return largest_component_mask(mask_t.numpy()).astype(bool)

    cleaned = tf.py_function(func=_np_fn, inp=[mask], Tout=tf.bool)
    cleaned.set_shape(mask.shape)
    return cleaned


class CropForeground(KeyedTransform, InvertibleTransform):
    """Crop selected tensors to the foreground region of a source tensor.

    This transform detects foreground in ``source_key`` using ``select_fn``,
    builds a bounding box around that region, and applies the same crop to all
    selected tensors. It supports both 2D channel-last tensors ``(H, W, C)``
    and 3D channel-last tensors ``(D, H, W, C)``.

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
        mask_postprocess: Optional cleanup step applied to the foreground mask
            (after ``select_fn``, before the bounding box is computed). Accepts
            either a preset name (``"largest_component"``) or a custom callable
            that takes a boolean mask and returns a boolean mask of the same
            shape. Use this when ``select_fn`` alone over- or under-segments the
            foreground because it can only see pixel values, not spatial
            structure; for example when non-anatomical content (scanner text,
            laterality labels, markers) shares intensity with real foreground
            and gets pulled into the bounding box, inflating the crop.

        .. note::

            * **When to use it**: only when ``select_fn`` is producing a
                bounding box that includes content you don't want, or is
                missing content it should include. If ``select_fn`` alone
                gives a clean mask, leave this unset; it adds a mask-cleanup
                pass that isn't free.
            * **What it can do**: presets and custom callables operate purely
                on the boolean mask, so any spatial-only cleanup is fair game,
                discarding small disconnected regions, bridging fragmented
                regions, filling internal holes, restricting to
                border-touching regions, etc. Custom callables compose with
                the rest of ``CropForeground`` (``margin``, ``allow_smaller``,
                ``k_divisible``) exactly like the unmodified mask would.
            * **Limitations**: it only sees the mask, not the source image's
                pixel values or any other tensor in ``keys``. Tt cannot make
                decisions based on intensity, texture, or content elsewhere in
                the sample. It also cannot recover information that
                ``select_fn`` already discarded. Built-in presets requiring
                ``scipy`` are eager, host-side operations (bridged internally
                via ``tf.py_function`` on the TensorFlow backend), so they run
                per-sample rather than as traceable ops; fine for a
                ``tf.data`` input pipeline, not intended for use inside a
                model's forward pass. Preset defaults (e.g. bridging/closing
                strength) are tuned heuristics, not guarantees -- verify on a
                sample of your own data, since the right amount of bridging
                differs by modality (thin gaps at soft tissue edges in 2D
                mammography vs. tighter anatomy-to-artifact spacing in 3D CT).

        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Crop a 2D image-label pair using a raw Python dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import CropForeground

            transform = CropForeground(
                keys=["image", "label"],
                source_key="image",
                margin=4,
            )

            image = tf.pad(tf.ones((24, 24, 1)), paddings=[[8, 8], [8, 8], [0, 0]])
            label = tf.cast(image > 0, tf.float32)

            result = transform({"image": image, "label": label})
            cropped_image = result["image"]
            cropped_label = result["label"]

        Crop a 3D image volume using a ``TensorBundle`` and inspect the stored
        crop coordinates:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import CropForeground, TensorBundle

            transform = CropForeground(
                keys=["image"],
                margin=(2, 4, 4),
                k_divisible=2,
            )

            image = tf.pad(
                tf.ones((8, 16, 16, 1)),
                paddings=[[4, 4], [8, 8], [8, 8], [0, 0]],
            )
            bundle = TensorBundle({"image": image})

            result = transform(bundle)
            start = result["foreground_start_coord"]
            end = result["foreground_end_coord"]

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
        mask_postprocess: Optional[Union[str, Callable]] = None,
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
        self.mask_postprocess_fn = self._resolve_postprocess(mask_postprocess)

    def _resolve_postprocess(self, spec):
        if spec is None or callable(spec):
            return spec
        if spec == "keep_largest_connected_component":
            return largest_component_mask_tf
        raise ValueError(f"Unknown mask_postprocess preset: {spec!r}")

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        if self.source_key not in bundle.data:
            if self.allow_missing_keys:
                return bundle
            raise KeyError(f"Key '{self.source_key}' not found in input data.")

        source_data = bundle.data[self.source_key]
        spatial_rank = get_spatial_rank(source_data)
        image_shape = get_spatial_shape(source_data)

        if self.channel_indices is not None:
            source_data = tf.gather(source_data, self.channel_indices, axis=-1)

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
            allow_missing_keys=self.allow_missing_keys,
        )

        def apply_crop(tensor: tf.Tensor, key: str) -> tf.Tensor:
            original_shapes[key] = get_spatial_shape(tensor)
            return crop.crop_tensor(tensor, min_coords, crop_size)

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
            allow_missing_keys=self.allow_missing_keys,
        )

        def apply_inverse_crop(tensor: tf.Tensor, key: str) -> tf.Tensor:
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
        image: tf.Tensor,
        select_fn: Callable,
        spatial_rank: int,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Find the bounding box of the foreground in the image."""
        mask = tf.reduce_any(select_fn(image), axis=-1)

        if self.mask_postprocess_fn is not None:
            mask = self.mask_postprocess_fn(mask)

        coords = tf.where(mask)
        coord_dtype = coords.dtype

        def empty_bbox():
            return (
                tf.zeros((spatial_rank,), dtype=coord_dtype),
                tf.cast(get_spatial_shape(image), coord_dtype),
            )

        def foreground_bbox():
            min_coords = tf.cast(tf.reduce_min(coords[:, :spatial_rank], axis=0), coord_dtype)
            max_coords = tf.cast(tf.reduce_max(coords[:, :spatial_rank], axis=0) + 1, coord_dtype)
            return min_coords, max_coords

        return tf.cond(tf.shape(coords)[0] > 0, foreground_bbox, empty_bbox)

    def add_margin(
        self,
        min_coords: tf.Tensor,
        max_coords: tf.Tensor,
        margin: Union[Sequence[int], int],
        image_shape: tf.Tensor,
        allow_smaller: bool,
        spatial_rank: int,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Add margin to the bounding box while staying inside image bounds."""
        margin = tf.convert_to_tensor(
            ensure_spatial_tuple(margin, spatial_rank, "margin"),
            dtype=tf.int32,
        )

        min_coords = tf.maximum(tf.cast(min_coords, tf.int32) - margin, 0)
        max_coords = tf.minimum(
            tf.cast(max_coords, tf.int32) + margin, tf.cast(image_shape, tf.int32)
        )

        if not allow_smaller:
            min_coords = tf.minimum(min_coords, tf.cast(image_shape, tf.int32) - margin)
            max_coords = tf.maximum(max_coords, margin)

        return min_coords, max_coords

    def make_divisible(
        self,
        min_coords: tf.Tensor,
        max_coords: tf.Tensor,
        k_divisible: Union[Sequence[int], int],
        image_shape: tf.Tensor,
        spatial_rank: int,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Expand the bounding box so its size is divisible by ``k_divisible``."""
        k_divisible = tf.convert_to_tensor(
            ensure_spatial_tuple(k_divisible, spatial_rank, "k_divisible"),
            dtype=tf.int32,
        )

        size = max_coords - min_coords
        remainder = size % k_divisible
        padding = tf.where(remainder != 0, k_divisible - remainder, 0)
        max_coords = tf.minimum(max_coords + padding, tf.cast(image_shape, tf.int32))
        return min_coords, max_coords

    def _get_last_crop_foreground_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)
