from typing import Sequence, Tuple

import tensorflow as tf

from medicai.transforms.depth_interpolate import DepthInterpolation as depth_interpolate
from medicai.transforms.meta_tensor import MetaTensor


class Resize:
    def __init__(
        self,
        keys: Sequence[str],
        mode: Tuple[str, str] = ("bilinear", "nearest"),
        spatial_shape: Tuple[int, ...] = (96, 96, 96),
    ):
        self.keys = keys
        self.mode = dict(zip(keys, mode))
        self.spatial_shape = spatial_shape
        self.only_image = only_image

    def __call__(self, inputs: MetaTensor) -> MetaTensor:
        for key in self.keys:
            if key not in inputs.data:
                continue

            image = inputs.data[key]

            if len(self.spatial_shape) == 2:
                inputs.data[key] = self._resize_2d(image, key)
            elif len(self.spatial_shape) == 3:
                inputs.data[key] = self._resize_3d(image, key)
            else:
                raise ValueError("spatial_shape must be either 2D or 3D.")

        return inputs

    def _resize_2d(self, image: tf.Tensor, key: str) -> tf.Tensor:
        new_height, new_width = self.spatial_shape
        resized_hw = tf.image.resize(image, [new_height, new_width], method=self.mode.get(key))
        return resized_hw

    def _resize_3d(self, image: tf.Tensor, key: str) -> tf.Tensor:
        new_depth, new_height, new_width = self.spatial_shape
        resized_hw = tf.image.resize(image, [new_height, new_width], method=self.mode.get(key))
        resized_dhw = depth_interpolate()(
            resized_hw, new_depth, depth_axis=0, method=self.mode.get(key)
        )
        return resized_dhw
