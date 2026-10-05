from dataclasses import dataclass
import math

from keras import ops

from medicai.transforms import Compose, RandomAffine, RandomFlip


@dataclass
class AugmentationConfig:
    """Probability gates and parameters for nnU-Net-style transforms."""

    p_rotation: float = 0.2
    p_scale: float = 0.2
    # RandomElasticTransform exists, but official nnU-Net v2 disables it by default.
    p_elastic: float = 0.0
    # TODO(nnUNet-parity): add regular gamma (p=0.3), inverted gamma (p=0.1),
    # channel-wise retain-stats handling, and masking outside normalization regions.
    p_gamma: float = 0.3
    # TODO(nnUNet-parity): add Gaussian noise (p=0.1) and Gaussian blur (p=0.2,
    # including per-channel blur probability).
    p_noise: float = 0.1
    # TODO(nnUNet-parity): add multiplicative brightness (p=0.15), contrast
    # (p=0.15, preserve range), and simulated low resolution (p=0.25).
    p_mirror: float = 0.5
    rotation_angle_range: float = math.pi / 6  # Official isotropic 3D default: +/-30 degrees.
    scale_range: tuple[float, float] = (
        0.7,
        1.4,
    )  # Actual scales; RandomAffine takes offsets from 1.0.
    elastic_alpha: float = 300.0
    elastic_sigma: float = 14.0
    gamma_range = (0.7, 1.5)
    noise_variance = (0.0, 0.1)
    label_fill_value: int = 0
    # TODO(nnUNet-parity): add per-axis mirror configuration and reproduce
    # official operation ordering/probabilities for each 2D/3D configuration.
    mirror_axes = (0, 1, 2)


class AugmentationPipeline:
    """Apply a supervised augmentation chain using MedicAI's ``Compose``.

    Both image and label are required. ``jit_compile`` is forwarded to
    :class:`medicai.transforms.Compose` so the complete chain can be evaluated
    eagerly or compiled by the selected Keras backend.
    """

    def __init__(
        self,
        config=None,
        patch_size=None,
        spacing=None,
        seed=None,
        jit_compile: bool = False,
    ):
        if config is None:
            config = AugmentationConfig()
        self.config = config
        self.patch_size = tuple(int(size) for size in patch_size) if patch_size else (128, 128, 128)
        self.spacing = tuple(float(value) for value in spacing) if spacing is not None else None

        c = self.config
        keys = ["image", "label"]

        spatial_rank = len(self.patch_size)
        if spatial_rank not in (2, 3):
            raise ValueError("patch_size must describe either a 2D or 3D spatial patch.")
        if self.spacing is not None and len(self.spacing) != spatial_rank:
            raise ValueError("spacing must have one value per spatial patch axis.")
        self.input_layout = "HWC" if spatial_rank == 2 else "DHWC"

        self.rotation_factor = self._rotation_ranges(c)
        self.initial_patch_size = self._compute_initial_patch_size(c)

        # MedicAI scale factors are relative offsets from 1.0: nnU-Net's
        # actual [0.7, 1.4] range therefore maps to [-0.3, 0.4].
        scale_factor = (c.scale_range[0] - 1.0, c.scale_range[1] - 1.0)
        rotation_factor = self.rotation_factor if c.p_rotation > 0 else 0.0
        if c.p_scale <= 0:
            scale_factor = 0.0

        # TODO(nnUNet-parity): RandomAffine currently gates rotation and scale
        # together with one probability; nnU-Net samples their p=0.2 gates
        # independently. Keep this combined transform until separate component
        # gates can be added without requiring multiple interpolation passes.
        spatial = RandomAffine(
            keys=keys,
            rotation_factor=rotation_factor,
            scale_factor=scale_factor,
            prob=max(c.p_rotation, c.p_scale),
            interpolation={
                "image": "bilinear" if spatial_rank == 2 else "trilinear",
                "label": "nearest",
            },
            fill_mode="constant",
            fill_value={"image": 0.0, "label": c.label_fill_value},
            input_layout=self.input_layout,
            seed=seed,
        )
        flips = [
            RandomFlip(
                keys=keys,
                prob=c.p_mirror,
                spatial_axis=axis,
                input_layout=self.input_layout,
                seed=None if seed is None else seed + axis + 1,
            )
            for axis in range(spatial_rank)
        ]
        self.transform = Compose([spatial, *flips], jit_compile=jit_compile)

    def _rotation_ranges(self, config):
        """Select nnU-Net-style rotation axes from rank and target spacing."""
        rank = len(self.patch_size)
        if rank == 2:
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                angle = math.pi / 12
            else:
                angle = math.pi
            return {"z": (-angle, angle)}

        if self.spacing is not None and max(self.spacing) / min(self.spacing) > 3.0:
            # Approximate nnU-Net dummy-2D by rotating only in the plane normal
            # to the coarsest (usually through-plane) axis.
            coarse_axis = max(range(3), key=self.spacing.__getitem__)
            rotation_axis = ("z", "y", "x")[coarse_axis]
            return {rotation_axis: (-math.pi, math.pi)}

        angle = config.rotation_angle_range
        return {axis: (-angle, angle) for axis in ("z", "y", "x")}

    def _compute_initial_patch_size(self, config):
        """Conservatively enlarge the sampled patch for spatial resampling."""
        patch = self.patch_size
        if config.p_scale <= 0:
            min_scale = 1.0
        else:
            min_scale = min(config.scale_range)

        angles = {
            "z": max(abs(value) for value in self.rotation_factor.get("z", (0.0, 0.0))),
            "y": max(abs(value) for value in self.rotation_factor.get("y", (0.0, 0.0))),
            "x": max(abs(value) for value in self.rotation_factor.get("x", (0.0, 0.0))),
        }
        if config.p_rotation <= 0:
            angles = {axis: 0.0 for axis in angles}
        half = [size / 2.0 for size in patch]
        if len(patch) == 2:
            angle = angles["z"]
            c, s = abs(math.cos(angle)), abs(math.sin(angle))
            extent = [c * half[0] + s * half[1], s * half[0] + c * half[1]]
        else:
            extent = half[:]
            for angle, axes in (
                (angles["z"], (1, 2)),
                (angles["y"], (0, 2)),
                (angles["x"], (0, 1)),
            ):
                first, second = axes
                c, s = abs(math.cos(angle)), abs(math.sin(angle))
                rotated = extent[:]
                rotated[first] = c * half[first] + s * half[second]
                rotated[second] = s * half[first] + c * half[second]
                extent = [max(a, b) for a, b in zip(extent, rotated)]

        # Scale up the source patch so zoom-in still covers the final crop.
        return tuple(
            max(size, int(math.ceil(2 * value / min_scale)))
            for size, value in zip(patch, extent)
        )

    def __call__(
        self,
        image,
        label,
        patch_size=None,
        label_is_regions=False,
    ):

        # Convert to backend tensors
        image_tensor = ops.convert_to_tensor(image, dtype="float32")
        label_tensor = ops.convert_to_tensor(label, dtype="float32")
        label_without_channel = len(label_tensor.shape) == len(image_tensor.shape) - 1
        if label_without_channel:
            label_tensor = ops.expand_dims(label_tensor, axis=-1)

        result = self.transform({"image": image_tensor, "label": label_tensor})
        output_size = tuple(patch_size or self.patch_size)
        image_result = self._center_crop(result["image"], output_size)
        label_result = self._center_crop(result["label"], output_size)
        label_dtype = "float32" if label_is_regions else "int64"
        transformed_label = ops.cast(ops.round(label_result), label_dtype)
        if label_without_channel:
            transformed_label = ops.squeeze(transformed_label, axis=-1)
        return image_result, transformed_label

    @staticmethod
    def _center_crop(tensor, output_size):
        spatial_rank = len(output_size)
        input_shape = tuple(int(value) for value in tensor.shape[:spatial_rank])
        starts = [(source - target) // 2 for source, target in zip(input_shape, output_size)]
        slices = tuple(
            slice(start, start + target) for start, target in zip(starts, output_size)
        ) + (slice(None),)
        return tensor[slices]
