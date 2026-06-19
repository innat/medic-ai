from typing import Callable, Optional, Sequence, Union

import tensorflow as tf

from ..base import KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import ensure_spatial_tuple, get_spatial_rank, get_spatial_shape
from .spatial_crop import SpatialCrop


class CropForeground(KeyedTransform):
    """Crop selected tensors to the foreground region of a source tensor.

    This transform detects foreground in ``source_key`` using ``select_fn``,
    builds a bounding box around that region, and applies the same crop to all
    selected tensors. It supports both 2D channel-last tensors ``(H, W, C)``
    and 3D channel-last tensors ``(D, H, W, C)``.
    """

    def __init__(
        self,
        keys: Sequence[str] = ("image", "label"),
        source_key: str = "image",
        select_fn: Callable = lambda x: x > 0,
        channel_indices: Optional[Sequence[int]] = None,
        margin: Union[Sequence[int], int] = 0,
        allow_smaller: bool = True,
        k_divisible: Union[Sequence[int], int] = 1,
        start_coord_key: Optional[str] = "foreground_start_coord",
        end_coord_key: Optional[str] = "foreground_end_coord",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.source_key = source_key
        self.select_fn = select_fn
        self.channel_indices = channel_indices
        self.margin = margin
        self.allow_smaller = allow_smaller
        self.k_divisible = k_divisible
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key

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

        roi_size = max_coords - min_coords
        crop = SpatialCrop(
            keys=self.keys,
            roi_size=1,
            allow_missing_keys=self.allow_missing_keys,
        )
        present_keys = crop.apply_to_present_keys(
            bundle,
            lambda tensor, _: crop.crop_tensor(tensor, min_coords, roi_size),
        )

        if self.start_coord_key is not None:
            bundle.meta[self.start_coord_key] = min_coords
        if self.end_coord_key is not None:
            bundle.meta[self.end_coord_key] = max_coords

        self.record_transform(
            bundle,
            {
                "keys": list(present_keys),
                "roi_start": min_coords,
                "roi_size": roi_size,
                "source_key": self.source_key,
            },
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
        coords = tf.where(mask)

        def empty_bbox():
            return tf.zeros((spatial_rank,), dtype=tf.int32), get_spatial_shape(image)

        def foreground_bbox():
            min_coords = tf.reduce_min(coords[:, :spatial_rank], axis=0)
            max_coords = tf.reduce_max(coords[:, :spatial_rank], axis=0) + 1
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
        max_coords = tf.minimum(tf.cast(max_coords, tf.int32) + margin, tf.cast(image_shape, tf.int32))

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
