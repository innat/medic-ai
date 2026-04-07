import json
import random
from pathlib import Path

import keras
import numpy as np
from scipy.ndimage import zoom as ndimage_zoom

from medicai.trainer.nnunet.utils.io import load_npz


class nnUNetDataset(keras.utils.PyDataset):
    """
    Keras 3 PyDataset for loading and augmenting nnU-Net batches on the fly.
    """

    def __init__(
        self,
        case_files,
        batch_size,
        patch_size,
        augmentor,
        train_cfg,
        net_cfg,
        task_type="multi-class",
        augment=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.case_files = case_files
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.augmentor = augmentor
        self.train_cfg = train_cfg
        self.net_cfg = net_cfg
        self.task_type = task_type
        self.augment = augment

        self.properties_map = {}
        for cf in self.case_files:
            cf_path = Path(cf)
            prop_file = cf_path.parent / "properties" / f"{cf_path.stem}.json"
            if prop_file.exists():
                try:
                    with open(prop_file, "r", encoding="utf-8") as f:
                        self.properties_map[str(cf)] = json.load(f)
                except Exception:
                    pass

    def __len__(self):
        # nnU-Net defines epoch length directly in config, independent of actual dataset size
        return self.train_cfg.iters_per_epoch

    def __getitem__(self, index):
        # We sample randomly from the entire dataset for `batch_size` items
        # nnU-Net traditionally randomly samples indefinitely per iter.
        batch_images = []
        batch_labels = []

        while len(batch_images) < self.batch_size:
            case_file = random.choice(self.case_files)
            data = load_npz(case_file)
            image = data["data"]
            label = data.get("seg", None)

            # Preprocessed .npz files are always saved in channel-last format (D, H, W, C)
            image_cl = image
            label_cl = label

            props = self.properties_map.get(str(case_file))
            spatial_shape = image_cl.shape[:-1]

            # Determine if we should force a foreground class
            if props and self.train_cfg.use_fg_oversampling:
                force_fg = random.random() < (1.0 / 3.0)
            else:
                force_fg = False

            patch_center = None
            if force_fg and props and "class_locations" in props and props["class_locations"]:
                available_classes = list(props["class_locations"].keys())
                if available_classes:
                    chosen_class = random.choice(available_classes)
                    locations = props["class_locations"][chosen_class]
                    if locations:
                        patch_center = random.choice(locations)

            if patch_center is None:
                patch_center = []
                for dim_size in spatial_shape:
                    patch_center.append(random.randint(0, dim_size - 1))

            # Calculate bbox slices for cropping
            bbox_slices = []
            pad_before = []
            pad_after = []
            for ax, center in enumerate(patch_center):
                ps = self.patch_size[ax]
                # Try to center the patch normally
                start = center - ps // 2
                end = start + ps

                # Handle boundaries
                if start < 0 or end > spatial_shape[ax]:
                    pb = max(0, -start)
                    pa = max(0, end - spatial_shape[ax])
                    start_c = max(0, start)
                    end_c = min(spatial_shape[ax], end)
                else:
                    pb = pa = 0
                    start_c = start
                    end_c = end

                bbox_slices.append(slice(start_c, end_c))
                pad_before.append(pb)
                pad_after.append(pa)

            patch_image = image_cl[tuple(bbox_slices)]
            pad_width_img = [(pb, pa) for pb, pa in zip(pad_before, pad_after)] + [(0, 0)]
            if any(p > 0 for p in pad_before + pad_after):
                patch_image = np.pad(patch_image, pad_width_img, mode="constant", constant_values=0)
            image_cl = patch_image

            if label_cl is not None:
                patch_label = label_cl[tuple(bbox_slices)]
                pad_width_lbl = [(pb, pa) for pb, pa in zip(pad_before, pad_after)]
                if label_cl.ndim > len(spatial_shape):
                    pad_width_lbl += [(0, 0)]
                if any(p > 0 for p in pad_before + pad_after):
                    patch_label = np.pad(
                        patch_label, pad_width_lbl, mode="constant", constant_values=0
                    )
                label_cl = patch_label

            if self.augment:
                if (
                    label_cl is not None
                    and self.task_type != "multi-label"
                    and label_cl.ndim == len(image_cl.shape) - 1
                ):
                    label_cl = label_cl[..., np.newaxis]

                image_cl, label_cl = self.augmentor(
                    image_cl,
                    label_cl,
                    patch_size=self.patch_size,
                )

                if (
                    label_cl is not None
                    and self.task_type != "multi-label"
                    and label_cl.ndim == image_cl.ndim
                ):
                    label_cl = np.squeeze(label_cl, axis=-1)

            batch_images.append(np.asarray(image_cl, dtype=np.float32))

            if label_cl is None:
                shape = batch_images[-1].shape[:-1] if hasattr(batch_images[-1], "shape") else []
                batch_labels.append(np.zeros(shape, dtype=np.int64))
            else:
                label_dtype = np.float32 if self.task_type == "multi-label" else np.int64
                batch_labels.append(np.asarray(label_cl, dtype=label_dtype))

        image_batch = np.stack(batch_images, axis=0)
        label_batch = np.stack(batch_labels, axis=0)

        if self.train_cfg.deep_supervision and self.net_cfg and self.net_cfg.deep_supervision:
            y_dict = {"final": label_batch}
            pool_kernels = getattr(self.net_cfg, "pool_op_kernel_sizes", None)
            for i in range(self.net_cfg.n_pooling - 1):
                # Compute cumulative downsampling factor for this auxiliary level
                # Level i corresponds to (i+1) pooling operations from full res
                if pool_kernels and len(pool_kernels) > i:
                    # Use actual pool kernel sizes for per-axis factors
                    zoom_factors = [1.0]  # batch axis
                    for ax in range(len(self.patch_size)):
                        factor = 1.0
                        for level in range(i + 1):
                            if level < len(pool_kernels):
                                ax_idx = min(ax, len(pool_kernels[level]) - 1)
                                factor /= pool_kernels[level][ax_idx]
                        zoom_factors.append(factor)
                else:
                    # Fallback: isotropic 2x downsampling per level
                    scale = 0.5 ** (i + 1)
                    zoom_factors = [1.0] + [scale] * len(self.patch_size)

                # Downsample the label batch using nearest-neighbor
                ds_label = ndimage_zoom(
                    label_batch.astype(np.float32),
                    zoom_factors,
                    order=0,
                    mode="nearest",
                ).astype(label_batch.dtype)
                y_dict[f"aux_{i}"] = ds_label
            return image_batch, y_dict

        return image_batch, label_batch
