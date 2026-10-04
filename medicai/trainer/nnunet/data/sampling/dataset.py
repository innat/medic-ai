"""nnU-Net preprocessed-case patch dataset."""

import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import zoom as ndimage_zoom


class nnUNetDataset:
    """Sample channel-last patches from MedicAI nnU-Net preprocessed cases.

    A random case is selected for each batch position. Foreground oversampling
    follows nnU-Net's deterministic last-fraction rule within each batch, and
    foreground coordinates are loaded from each case's properties JSON rather
    than recomputed from its segmentation.

    Args:
        case_files: Paths to preprocessed ``.npz`` cases containing ``image`` and
            ``label`` arrays. Images use ``DHWC`` layout; labels use
            ``DHW`` or ``DHWR`` for categorical or region targets.
        batch_size: Planned number of patches per batch.
        patch_size: Initial spatial size sampled before augmentation. Two-dimensional
            patches are sampled from 3D cases one depth slice at a time.
        augmentor: Optional callable that transforms an image/label patch pair.
        train_cfg: Training configuration containing ``iters_per_epoch`` and
            optionally ``use_fg_oversampling`` and ``deep_supervision``.
        net_cfg: Selected network configuration, including pooling kernels.
        task_type: ``"binary"``, ``"multi-class"``, or ``"region_based"``.
        augment: Whether to apply the supplied augmentor.
        final_patch_size: Network input patch size. Defaults to ``patch_size``.
        oversample_foreground_percent: Fraction of batch slots reserved for
            foreground-centered crops, defaulting to nnU-Net's ``0.33``.
        ignore_class_ids: Source label IDs treated as ignored by preprocessing.
        case_cache_size: Number of decompressed cases retained in an LRU cache.

    Returns:
        Batches are ``(image_batch, label_batch)`` pairs. With deep supervision
        enabled, targets are a mapping keyed like the model outputs.
    """

    def __init__(
        self,
        case_files: Sequence[str | Path],
        batch_size: int,
        patch_size: Sequence[int],
        augmentor: Any,
        train_cfg: Any,
        net_cfg: Any,
        task_type: str = "multi_class",
        augment: bool = True,
        final_patch_size: Sequence[int] | None = None,
        oversample_foreground_percent: float = 0.33,
        ignore_class_ids: Sequence[int] | None = None,
        case_cache_size: int = 1,
    ) -> None:
        self.case_files = [Path(path) for path in case_files]
        self.batch_size = int(batch_size)
        self.patch_size = tuple(int(size) for size in patch_size)
        selected_final_patch_size = self.patch_size if final_patch_size is None else final_patch_size
        self.final_patch_size = tuple(int(size) for size in selected_final_patch_size)
        self.augmentor = augmentor
        self.train_cfg = train_cfg
        self.net_cfg = net_cfg
        self.task_type = task_type
        self.augment = augment
        self.oversample_foreground_percent = float(oversample_foreground_percent)
        self.ignore_class_ids = {int(value) for value in (ignore_class_ids or [])}
        self.case_cache_size = int(case_cache_size)
        self._case_cache: OrderedDict[Path, tuple[np.ndarray, np.ndarray | None]] = OrderedDict()

        if not self.case_files:
            raise ValueError("Cannot sample nnU-Net batches from an empty case list.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if len(self.patch_size) not in (2, 3) or len(self.patch_size) != len(self.final_patch_size):
            raise ValueError("patch_size and final_patch_size must have matching 2D or 3D ranks.")
        if any(size < 1 for size in self.patch_size + self.final_patch_size):
            raise ValueError("Patch dimensions must be positive.")
        if not 0.0 <= self.oversample_foreground_percent <= 1.0:
            raise ValueError("oversample_foreground_percent must be between 0 and 1.")
        if self.case_cache_size < 0:
            raise ValueError("case_cache_size cannot be negative.")

        self.properties_map = {}
        for case_file in self.case_files:
            properties_path = case_file.parent / "properties" / f"{case_file.stem}.json"
            if properties_path.is_file():
                with properties_path.open(encoding="utf-8") as stream:
                    self.properties_map[case_file] = json.load(stream)

    def __len__(self) -> int:
        """Return the configured number of batches per epoch."""
        return int(self.train_cfg.iters_per_epoch)

    def _load_case(self, case_file: Path) -> tuple[np.ndarray, np.ndarray]:
        if case_file in self._case_cache:
            self._case_cache.move_to_end(case_file)
            return self._case_cache[case_file]

        with np.load(case_file, allow_pickle=False) as archive:
            if "image" not in archive:
                raise ValueError(f"Preprocessed case {case_file} has no 'image' array.")
            if "label" not in archive:
                raise ValueError(
                    f"Training case {case_file} has no 'label' array; nnU-Net training requires labels."
                )
            image = np.array(archive["image"], dtype=np.float32, copy=True)
            label = np.array(archive["label"], copy=True)

        case = (image, label)
        if self.case_cache_size:
            self._case_cache[case_file] = case
            self._case_cache.move_to_end(case_file)
            while len(self._case_cache) > self.case_cache_size:
                self._case_cache.popitem(last=False)
        return case

    @staticmethod
    def _crop_and_pad(
        array: np.ndarray,
        lower: Sequence[int],
        patch_size: Sequence[int],
        value: int,
    ) -> np.ndarray:
        spatial_rank = len(patch_size)
        spatial_shape = array.shape[:spatial_rank]
        upper = [start + size for start, size in zip(lower, patch_size, strict=True)]
        source_lower = [max(0, start) for start in lower]
        source_upper = [min(dim, end) for dim, end in zip(spatial_shape, upper, strict=True)]
        slices = tuple(slice(start, end) for start, end in zip(source_lower, source_upper))
        cropped = array[slices + (slice(None),) * (array.ndim - spatial_rank)]
        pad_width = [
            (max(0, -start), max(0, end - dim))
            for start, end, dim in zip(lower, upper, spatial_shape, strict=True)
        ]
        pad_width.extend([(0, 0)] * (array.ndim - spatial_rank))
        if any(before or after for before, after in pad_width):
            cropped = np.pad(cropped, pad_width, mode="constant", constant_values=value)
        return cropped

    def _sample_patch(
        self, rng: Any, case_file: Path, slot: int
    ) -> tuple[np.ndarray, np.ndarray]:
        image, label = self._load_case(case_file)
        if image.ndim != 4:
            raise ValueError(f"Expected channel-last DHWC preprocessed image, got {image.shape}.")
        if label.shape[:3] != image.shape[:3]:
            raise ValueError(f"Image and label spatial shapes differ for case {case_file}.")

        spatial_rank = len(self.patch_size)
        # nnU-Net's 2D configuration still reads a 3D case, then samples a slice.
        is_2d_from_3d = spatial_rank == 2
        requested_patch_size = self.patch_size if self.augment else self.final_patch_size
        patch_size = (1, *requested_patch_size) if is_2d_from_3d else requested_patch_size
        final_patch_size = (
            (1, *self.final_patch_size) if is_2d_from_3d else self.final_patch_size
        )
        source_shape = image.shape[: len(patch_size)]
        need_to_pad = [
            patch - final for patch, final in zip(patch_size, final_patch_size, strict=True)
        ]
        for axis, size in enumerate(source_shape):
            if size + need_to_pad[axis] < patch_size[axis]:
                need_to_pad[axis] = patch_size[axis] - size

        # Match nnU-Net's floor division for odd padding amounts.
        lower_bounds = [-padding // 2 for padding in need_to_pad]
        upper_bounds = [
            size + padding // 2 + padding % 2 - patch
            for size, padding, patch in zip(source_shape, need_to_pad, patch_size, strict=True)
        ]

        oversample = bool(getattr(self.train_cfg, "use_fg_oversampling", True))
        force_foreground = oversample and slot >= round(
            self.batch_size * (1.0 - self.oversample_foreground_percent)
        )
        properties = self.properties_map.get(case_file, {})
        class_locations = properties.get("class_locations", {})
        locations_by_class = {
            key: locations for key, locations in class_locations.items() if locations
        }

        if force_foreground and locations_by_class:
            class_key = random.choice(list(locations_by_class))
            location = random.choice(locations_by_class[class_key])
            if len(location) != 3:
                raise ValueError(f"Expected DHW foreground coordinates, got {location} in {case_file}.")
            if is_2d_from_3d:
                depth, row, column = (int(value) for value in location)
                source_shape = image.shape[1:3]
                patch_size = requested_patch_size
                lower_bounds = [-need_to_pad[1] // 2, -need_to_pad[2] // 2]
                upper_bounds = [
                    source_shape[axis]
                    + need_to_pad[axis + 1] // 2
                    + need_to_pad[axis + 1] % 2
                    - patch_size[axis]
                    for axis in range(2)
                ]
                image = image[depth]
                label = label[depth]
                center = (row, column)
            else:
                center = tuple(int(value) for value in location)
            lower = [
                max(low, coordinate - patch // 2)
                for low, coordinate, patch in zip(lower_bounds, center, patch_size, strict=True)
            ]
        else:
            lower = [
                rng.randint(low, high)
                for low, high in zip(lower_bounds, upper_bounds, strict=True)
            ]
            if is_2d_from_3d:
                depth = rng.randrange(image.shape[0])
                lower = lower[1:]
                image = image[depth]
                label = label[depth]
                patch_size = requested_patch_size

        image_patch = self._crop_and_pad(image, lower, patch_size, value=0).astype(
            np.float32, copy=False
        )
        label_pad_value = min(self.ignore_class_ids) if self.ignore_class_ids else 0
        label_patch = self._crop_and_pad(label, lower, patch_size, value=label_pad_value)
        if self.augment and self.augmentor is not None:
            image_patch, label_patch = self.augmentor(
                image_patch,
                label_patch,
                patch_size=self.final_patch_size,
                label_is_regions=self.task_type == "region_based",
            )
            image_patch = np.asarray(image_patch, dtype=np.float32)
            label_patch = np.asarray(label_patch)

        # TODO(nnUNet-parity): ensure spatial augmentation fill values preserve
        # the configured integer ignore ID instead of replacing it with background.

        return image_patch, label_patch

    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray | dict[str, np.ndarray]]:
        """Sample one batch of patches; ``index`` identifies a batch slot in the epoch."""
        batch_images = []
        batch_labels = []
        for slot in range(self.batch_size):
            rng = random
            case_file = rng.choice(self.case_files)
            image, label = self._sample_patch(rng, case_file, slot)
            batch_images.append(image)
            batch_labels.append(label)

        image_batch = np.stack(batch_images, axis=0).astype(np.float32, copy=False)
        label_dtype = np.float32 if self.task_type == "region_based" else np.int64
        label_batch = np.stack(batch_labels, axis=0).astype(label_dtype, copy=False)

        if self.train_cfg.deep_supervision and self.net_cfg and self.net_cfg.deep_supervision:
            return image_batch, self._make_deep_supervision_targets(label_batch)
        return image_batch, label_batch

    def _make_deep_supervision_targets(self, labels: np.ndarray) -> dict[str, np.ndarray]:
        """Downsample labels with nearest-neighbor scales derived from pool kernels."""
        targets = {"final": labels}
        pool_kernels = getattr(self.net_cfg, "pool_op_kernel_sizes", None) or []
        cumulative = np.ones(len(self.patch_size), dtype=np.float64)
        for index in range(self.net_cfg.n_pooling - 1):
            if pool_kernels and index < len(pool_kernels):
                cumulative *= np.asarray(pool_kernels[index], dtype=np.float64)
                scales = [1.0, *[1.0 / value for value in cumulative]]
            else:
                scales = [1.0, *([0.5 ** (index + 1)] * len(self.patch_size))]
            if labels.ndim == len(self.patch_size) + 2:
                scales.append(1.0)
            targets[f"aux_{index}"] = ndimage_zoom(
                labels,
                scales,
                order=0,
                mode="nearest",
                prefilter=False,
            ).astype(labels.dtype, copy=False)
        return targets

class PatchBatchSource:
    """Expose an nnU-Net patch dataset as a Grain random-access source."""

    def __init__(self, patch_dataset: nnUNetDataset) -> None:
        self.patch_dataset = patch_dataset

    def __len__(self) -> int:
        return len(self.patch_dataset)

    def __getitem__(self, index: int) -> Any:
        return self.patch_dataset[index]


def build_pygrain_dataset(
    patch_dataset: nnUNetDataset,
    *,
    shuffle: bool = False,
    seed: int = 0,
    num_threads: int = 4,
) -> Any:
    """Wrap complete nnU-Net patch batches in a PyGrain iterator.

    Args:
        patch_dataset: Random-access dataset yielding complete batches.
        shuffle: Whether to shuffle batch order.
        seed: Grain shuffle seed.
        num_threads: Number of Grain reader threads.

    Returns:
        PyGrain iterator yielding ``(image_batch, label_batch)`` pairs.

    Raises:
        ImportError: If the optional ``grain`` package is not installed.
        ValueError: If ``num_threads`` is not positive.
    """
    if num_threads < 1:
        raise ValueError("num_threads must be positive.")
    try:
        import grain.python as pygrain
    except ImportError as exc:
        raise ImportError(
            "PyGrain is required for nnU-Net training inputs. Install it with "
            "`pip install 'medicai[nnunet]'` or `pip install grain`."
        ) from exc

    dataset = pygrain.MapDataset.source(PatchBatchSource(patch_dataset))
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    return dataset.to_iter_dataset(
        read_options=pygrain.ReadOptions(num_threads=num_threads)
    )


__all__ = ["nnUNetDataset", "PatchBatchSource", "build_pygrain_dataset"]
