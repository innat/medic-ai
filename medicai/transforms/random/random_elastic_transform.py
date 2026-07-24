import itertools
from typing import Sequence

import tensorflow as tf

from ..base import RandomTransform, _normalize_keys
from ..tensor_bundle import TensorBundle
from ..utils import get_spatial_rank


def _gaussian_kernel_1d(sigma: float, radius: int, dtype: tf.dtypes.DType) -> tf.Tensor:
    """Build a normalized 1D Gaussian kernel."""
    x = tf.range(-radius, radius + 1, dtype=dtype)
    sigma = tf.cast(sigma, dtype)
    kernel = tf.exp(-0.5 * tf.square(x / sigma))
    return kernel / tf.reduce_sum(kernel)


def _smooth_along_axis(
    x: tf.Tensor,
    kernel_1d: tf.Tensor,
    axis: int,
    radius: int,
    spatial_rank: int,
) -> tf.Tensor:
    """Apply one separable smoothing pass along one spatial axis."""
    kernel_shape = [1] * spatial_rank
    kernel_shape[axis] = int(kernel_1d.shape[0])
    kernel = tf.reshape(kernel_1d, kernel_shape + [1, 1])

    paddings = [[0, 0]] * (spatial_rank + 2)
    paddings[axis + 1] = [radius, radius]
    x = tf.pad(x, paddings, mode="SYMMETRIC")
    return tf.nn.convolution(x, kernel, padding="VALID")


def gaussian_smooth_nd(field: tf.Tensor, sigma: float, spatial_rank: int) -> tf.Tensor:
    """Apply separable Gaussian smoothing to a batched vector field."""
    dtype = field.dtype
    radius = max(1, int(round(3.0 * float(sigma))))
    kernel_1d = _gaussian_kernel_1d(sigma, radius, dtype)

    shape = tf.shape(field)
    batch_size = shape[0]
    spatial_shape = [shape[i + 1] for i in range(spatial_rank)]
    channels = shape[spatial_rank + 1]

    perm = [0, spatial_rank + 1] + list(range(1, spatial_rank + 1))
    x = tf.transpose(field, perm)
    x = tf.reshape(x, tf.concat([[batch_size * channels], spatial_shape, [1]], axis=0))

    for axis in range(spatial_rank):
        x = _smooth_along_axis(x, kernel_1d, axis, radius, spatial_rank)

    x = tf.reshape(x, tf.concat([[batch_size, channels], spatial_shape], axis=0))
    inv_perm = [0] + list(range(2, spatial_rank + 2)) + [1]
    return tf.transpose(x, inv_perm)


def random_displacement_field(
    spatial_shape: tf.Tensor,
    alpha: float,
    sigma: float,
    spatial_rank: int,
) -> tf.Tensor:
    """Sample and smooth one displacement field shaped ``(1, *spatial, R)``."""
    noise_shape = tf.concat([[1], spatial_shape, [spatial_rank]], axis=0)
    field = tf.random.normal(noise_shape, dtype=tf.float32)
    field = gaussian_smooth_nd(field, sigma=sigma, spatial_rank=spatial_rank)
    return field * tf.cast(alpha, field.dtype)


def build_sampling_grid(spatial_shape: tf.Tensor, spatial_rank: int) -> tf.Tensor:
    """Create a base grid shaped ``(1, *spatial, R)``."""
    ranges = [tf.range(spatial_shape[i]) for i in range(spatial_rank)]
    mesh = tf.meshgrid(*ranges, indexing="ij")
    grid = tf.cast(tf.stack(mesh, axis=-1), tf.float32)
    return grid[None, ...]


def nlinear_sample(volume: tf.Tensor, coords: tf.Tensor, spatial_rank: int) -> tf.Tensor:
    """Sample a batched image tensor using N-linear interpolation."""
    original_dtype = volume.dtype
    working_volume = tf.cast(volume, tf.float32)
    shape = tf.shape(working_volume)
    batch_size = shape[0]
    channels = shape[-1]
    spatial_sizes = [shape[i + 1] for i in range(spatial_rank)]
    spatial_sizes_f = [tf.cast(size, tf.float32) for size in spatial_sizes]

    floor_coords = [tf.floor(coords[..., i]) for i in range(spatial_rank)]
    frac_weights = [(coords[..., i] - floor_coords[i])[..., None] for i in range(spatial_rank)]

    index_shape = tf.shape(floor_coords[0])
    batch_index = tf.reshape(tf.range(batch_size), [batch_size] + [1] * spatial_rank)
    batch_index = tf.broadcast_to(batch_index, index_shape)

    output = tf.zeros(tf.concat([index_shape, [channels]], axis=0), dtype=tf.float32)

    for corner in itertools.product((0, 1), repeat=spatial_rank):
        indices = []
        weight = tf.ones_like(frac_weights[0])
        for axis, bit in enumerate(corner):
            coord = tf.clip_by_value(
                floor_coords[axis] + bit,
                0.0,
                spatial_sizes_f[axis] - 1.0,
            )
            indices.append(tf.cast(coord, tf.int32))
            weight = weight * (frac_weights[axis] if bit == 1 else (1.0 - frac_weights[axis]))
        gather_index = tf.stack([batch_index] + indices, axis=-1)
        corner_value = tf.gather_nd(working_volume, gather_index)
        output = output + corner_value * weight

    return tf.cast(output, original_dtype)


def nearest_sample(volume: tf.Tensor, coords: tf.Tensor, spatial_rank: int) -> tf.Tensor:
    """Sample a batched tensor using nearest-neighbor interpolation."""
    shape = tf.shape(volume)
    batch_size = shape[0]
    spatial_sizes = [tf.cast(shape[i + 1], coords.dtype) for i in range(spatial_rank)]

    indices = []
    for axis in range(spatial_rank):
        coord = tf.round(coords[..., axis])
        coord = tf.clip_by_value(coord, 0.0, spatial_sizes[axis] - 1.0)
        indices.append(tf.cast(coord, tf.int32))

    batch_index = tf.reshape(tf.range(batch_size), [batch_size] + [1] * spatial_rank)
    batch_index = tf.broadcast_to(batch_index, tf.shape(indices[0]))
    gather_index = tf.stack([batch_index] + indices, axis=-1)
    return tf.gather_nd(volume, gather_index)


class RandomElasticTransform(RandomTransform):
    """Randomly warp 2D images or 3D volumes with elastic deformation.

    ``RandomElasticTransform`` samples a smooth displacement field and uses
    it to warp the selected tensors. The first key is treated like an image and
    uses N-linear interpolation, while any additional keys are treated like
    masks/labels and use nearest-neighbor interpolation so label values remain
    valid.

    This transform supports:

    - 2D tensors shaped ``(H, W, C)``
    - 3D tensors shaped ``(D, H, W, C)``

    Args:
        keys: One or more keys to deform. The first key is interpolated with
            N-linear sampling; remaining keys use nearest-neighbor sampling.
        alpha: Displacement magnitude measured in pixels or voxels. Larger
            values produce stronger deformations.
        sigma: Standard deviation of the Gaussian smoothing applied to the
            displacement field. Larger values produce broader, smoother warps.
        prob: Probability of applying the deformation.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Randomly deform a 2D image and label:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomElasticTransform

            transform = RandomElasticTransform(
                keys=["image", "label"],
                alpha=8.0,
                sigma=3.0,
                prob=0.5,
            )
            image = tf.random.normal((64, 64, 1))
            label = tf.cast(image > 0, tf.int32)
            result = transform({"image": image, "label": label})
            print(result["image"].shape)

        Randomly deform a 3D image stored in a ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomElasticTransform, TensorBundle

            transform = RandomElasticTransform(keys=["image"], alpha=6.0, sigma=2.5, prob=0.5)
            image = tf.random.normal((24, 48, 48, 1))
            bundle = TensorBundle({"image": image})
            result = transform(bundle)
            print(result["image"].shape)
    """

    def __init__(
        self,
        keys: Sequence[str],
        alpha: float = 8.0,
        sigma: float = 3.0,
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob)
        self.keys = _normalize_keys(keys)
        if alpha < 0:
            raise ValueError(f"`alpha` must be non-negative. Received {alpha}.")
        if sigma <= 0:
            raise ValueError(f"`sigma` must be strictly positive. Received {sigma}.")
        self.alpha = alpha
        self.sigma = sigma
        self.allow_missing_keys = allow_missing_keys

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        present_keys = []
        for key in self.keys:
            if key in bundle.data:
                present_keys.append(key)
            elif not self.allow_missing_keys:
                raise KeyError(f"Key '{key}' not found in input data.")

        if not present_keys:
            return bundle

        reference = bundle.data[present_keys[0]]
        spatial_rank = get_spatial_rank(reference)
        if spatial_rank not in (2, 3):
            raise ValueError(
                f"{type(self).__name__} supports only 2D or 3D channel-last tensors; got "
                f"spatial rank {spatial_rank} for shape {reference.shape}."
            )

        reference_spatial_shape = tf.shape(reference)[:spatial_rank]
        for key in present_keys[1:]:
            tf.debugging.assert_equal(
                tf.shape(bundle.data[key])[:spatial_rank],
                reference_spatial_shape,
                message="All selected tensors must share the same spatial shape.",
            )

        should_apply = self.sample_should_apply()
        displacement = random_displacement_field(
            spatial_shape=reference_spatial_shape,
            alpha=self.alpha,
            sigma=self.sigma,
            spatial_rank=spatial_rank,
        )
        coords = build_sampling_grid(reference_spatial_shape, spatial_rank) + displacement

        for key in present_keys:
            tensor = bundle.data[key]
            bundle.data[key] = tf.cond(
                should_apply,
                lambda tensor=tensor, key=key: self.warp_tensor(
                    tensor,
                    key=key,
                    coords=coords,
                    spatial_rank=spatial_rank,
                    image_key=present_keys[0],
                ),
                lambda tensor=tensor: tensor,
            )

        self.record_random_transform(
            bundle,
            params={
                "keys": list(present_keys),
                "alpha": self.alpha,
                "sigma": self.sigma,
            },
            applied=should_apply,
            kernel="elastic_deformation",
        )
        return bundle

    def warp_tensor(
        self,
        tensor: tf.Tensor,
        *,
        key: str,
        coords: tf.Tensor,
        spatial_rank: int,
        image_key: str,
    ) -> tf.Tensor:
        """Warp one sample tensor using shared coordinates."""
        batched = tensor[None, ...]
        if key == image_key:
            return nlinear_sample(batched, coords, spatial_rank=spatial_rank)[0]
        return nearest_sample(batched, coords, spatial_rank=spatial_rank)[0]
