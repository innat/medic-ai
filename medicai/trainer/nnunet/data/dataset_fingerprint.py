from pathlib import Path

import numpy as np
from tqdm import tqdm

from medicai.trainer.nnunet.data.normalization import compute_intensity_stats
from medicai.trainer.nnunet.utils.config import DatasetFingerprint
from medicai.trainer.nnunet.utils.io import (
    collapse_single_channel,
    ensure_spacing,
    get_spatial_shape,
    infer_spatial_dims,
    load_medical_image,
    normalize_layout,
)


def _is_ct_modality(modality_name):
    return "ct" in modality_name.lower()


def _resolve_output_channels(task_type, class_names, target_class_ids):
    if task_type == "binary":
        return 1
    if task_type == "multi-label":
        if target_class_ids:
            return len(target_class_ids)
        return max(1, len(class_names) - 1)
    return len(class_names)


def _resolve_multilabel_regions(item, manifest):
    return item.regions or manifest.regions


def _resolve_multilabel_output_channels(item, manifest):
    if (item.label_output or manifest.label_output) == "regions":
        return len(_resolve_multilabel_regions(item, manifest))
    target_class_ids = manifest.target_class_ids
    if target_class_ids:
        return len(target_class_ids)
    return max(1, len(manifest.class_names) - 1)


def _collect_multilabel_class_stats(
    item, manifest, spatial_dims, ensure_channel_last, target_class_ids
):
    if item.labels is None:
        return {}, 0

    item_label_output = item.label_output or manifest.label_output
    label_layout = item.label_layout or manifest.label_layout
    counts = {}
    total_voxels = 0

    label_paths = item.labels if isinstance(item.labels, list) else [item.labels]
    if item_label_output == "regions":
        label_data, _, _, label_spacing = load_medical_image(
            label_paths[0],
            ensure_channel_last=ensure_channel_last,
        )
        label_dims = infer_spatial_dims(label_data, spacing=item.spacing or label_spacing)
        label_data = normalize_layout(
            label_data,
            label_dims,
            ensure_channel_last=ensure_channel_last,
            layout=label_layout,
        )
        label_data = collapse_single_channel(
            label_data, label_dims, ensure_channel_last=ensure_channel_last
        )
        regions = _resolve_multilabel_regions(item, manifest)
        for region_idx, class_ids in enumerate(regions, start=1):
            region_mask = np.isin(label_data.astype(np.int64), class_ids)
            counts[region_idx] = int(region_mask.sum())
        total_voxels = int(np.prod(get_spatial_shape(label_data, label_dims, ensure_channel_last)))
        return counts, total_voxels

    if len(label_paths) == 1:
        label_data, _, _, label_spacing = load_medical_image(
            label_paths[0],
            ensure_channel_last=ensure_channel_last,
        )
        label_dims = infer_spatial_dims(label_data, spacing=item.spacing or label_spacing)
        label_dims = max(label_dims, spatial_dims)
        label_data = normalize_layout(
            label_data,
            label_dims,
            ensure_channel_last=ensure_channel_last,
            layout=label_layout,
        )
        label_data = collapse_single_channel(
            label_data, label_dims, ensure_channel_last=ensure_channel_last
        )
        class_ids = (
            list(target_class_ids)
            if target_class_ids
            else list(range(1, int(label_data.max()) + 1))
        )
        for class_id in class_ids:
            counts[class_id] = int((label_data == class_id).sum())
        total_voxels = int(np.prod(get_spatial_shape(label_data, label_dims, ensure_channel_last)))
        return counts, total_voxels

    class_ids = list(target_class_ids) if target_class_ids else list(range(1, len(label_paths) + 1))
    for class_id, label_path in zip(class_ids, label_paths, strict=True):
        label_data, _, _, label_spacing = load_medical_image(
            label_path,
            ensure_channel_last=ensure_channel_last,
        )
        label_dims = infer_spatial_dims(label_data, spacing=label_spacing)
        label_dims = max(label_dims, spatial_dims)
        label_data = normalize_layout(
            label_data,
            label_dims,
            ensure_channel_last=ensure_channel_last,
            layout=label_layout,
        )
        label_data = collapse_single_channel(
            label_data, label_dims, ensure_channel_last=ensure_channel_last
        )
        counts[class_id] = counts.get(class_id, 0) + int((label_data > 0).sum())
        if total_voxels == 0:
            total_voxels = int(
                np.prod(get_spatial_shape(label_data, label_dims, ensure_channel_last))
            )

    return counts, total_voxels


def fingerprint_dataset(
    manifest,
    output_file=None,
    dataset_name=None,
    ensure_channel_last=True,
    max_cases=None,
):
    """
    Build a :class:`DatasetFingerprint` directly from a manifest.
    """

    modalities = manifest.modalities
    class_names = manifest.class_names
    task_type = manifest.task_type
    ignore_class_ids = manifest.ignore_class_ids
    target_class_ids = manifest.target_class_ids

    if max_cases is not None:
        items = manifest.items[:max_cases]
    else:
        items = manifest.items

    if not items:
        raise FileNotFoundError("No cases resolved inside manifest.")

    dataset_name = dataset_name or manifest.dataset_name
    n_classes = len(class_names)
    if task_type == "multi-label":
        output_channels = _resolve_multilabel_output_channels(items[0], manifest)
    else:
        output_channels = _resolve_output_channels(task_type, class_names, target_class_ids)

    spacings = []
    sizes = []
    all_images_per_modality = {i: [] for i in range(len(modalities))}
    class_voxel_counts = {c: 0 for c in range(n_classes)}
    total_voxels = 0
    spatial_dims = None

    for item in tqdm(items, desc="Fingerprinting"):
        case_files = [Path(p) for p in item.images]
        if not case_files:
            continue
        image_layout = item.image_layout or manifest.image_layout

        data_0, _, _, loaded_spacing = load_medical_image(
            case_files[0],
            ensure_channel_last=ensure_channel_last,
        )
        case_spatial_dims = infer_spatial_dims(
            data_0,
            spacing=item.spacing if item.spacing is not None else loaded_spacing,
        )
        data_0 = normalize_layout(
            data_0,
            case_spatial_dims,
            ensure_channel_last=ensure_channel_last,
            layout=image_layout,
        )
        spacing = ensure_spacing(
            item.spacing if item.spacing is not None else loaded_spacing,
            case_spatial_dims,
        )
        shape = get_spatial_shape(
            data_0,
            case_spatial_dims,
            ensure_channel_last=ensure_channel_last,
        )

        spacings.append([float(s) for s in spacing])
        sizes.append([int(d) for d in shape])
        spatial_dims = (
            case_spatial_dims if spatial_dims is None else max(spatial_dims, case_spatial_dims)
        )

        def _subsample(img):
            # Target ~10-15 voxels skip per dim; reduces memory by ~1000x for 3D
            if img.size > 100_000:
                slices = tuple(slice(0, None, 10) for _ in range(img.ndim))
                return img[slices]
            return img

        first_mod = collapse_single_channel(
            data_0, case_spatial_dims, ensure_channel_last=ensure_channel_last
        )
        if first_mod.ndim > case_spatial_dims:
            first_mod = first_mod[..., 0] if ensure_channel_last else first_mod[0]
        all_images_per_modality[0].append(_subsample(first_mod))

        for mod_idx in range(1, len(modalities)):
            if mod_idx >= len(case_files):
                continue
            image, _, _, image_spacing = load_medical_image(
                case_files[mod_idx],
                ensure_channel_last=ensure_channel_last,
            )
            image_dims = infer_spatial_dims(image, spacing=image_spacing)
            image = normalize_layout(
                image,
                image_dims,
                ensure_channel_last=ensure_channel_last,
                layout=image_layout,
            )
            image = collapse_single_channel(
                image, image_dims, ensure_channel_last=ensure_channel_last
            )
            if image.ndim > image_dims:
                image = image[..., 0] if ensure_channel_last else image[0]
            all_images_per_modality[mod_idx].append(_subsample(image))

        if item.labels is None:
            continue

        if task_type == "multi-label":
            multilabel_counts, case_total = _collect_multilabel_class_stats(
                item,
                manifest,
                case_spatial_dims,
                ensure_channel_last,
                target_class_ids,
            )
            for class_id, count in multilabel_counts.items():
                if class_id < len(class_names):
                    class_voxel_counts[class_id] += count
            total_voxels += case_total
        else:
            label_path = item.labels[0] if isinstance(item.labels, list) else item.labels
            label_data, _, _, label_spacing = load_medical_image(
                label_path,
                ensure_channel_last=ensure_channel_last,
            )
            label_dims = infer_spatial_dims(label_data, spacing=item.spacing or label_spacing)
            label_data = normalize_layout(
                label_data,
                label_dims,
                ensure_channel_last=ensure_channel_last,
                layout=item.label_layout or manifest.label_layout,
            )
            label_data = collapse_single_channel(
                label_data, label_dims, ensure_channel_last=ensure_channel_last
            )
            unique, counts = np.unique(label_data.astype(np.int64), return_counts=True)
            for val, count in zip(unique, counts):
                if int(val) in class_voxel_counts:
                    class_voxel_counts[int(val)] += int(count)
            total_voxels += int(
                np.prod(get_spatial_shape(label_data, label_dims, ensure_channel_last))
            )

    intensity_stats = {}
    is_ct_modality = [_is_ct_modality(name) for name in modalities]
    for mod_idx, mod_name in enumerate(modalities):
        imgs = all_images_per_modality.get(mod_idx, [])
        if not imgs:
            continue
        intensity_stats[mod_name] = compute_intensity_stats(
            imgs,
            nonzero_only=not is_ct_modality[mod_idx],
        )

    if not spacings:
        raise RuntimeError("No valid cases found during fingerprinting.")

    num_dims = len(spacings[0])
    median_spacing = [float(np.median([s[i] for s in spacings])) for i in range(num_dims)]
    median_size = [int(np.median([sz[i] for sz in sizes])) for i in range(num_dims)]

    if num_dims == 2:
        median_spacing = [1.0, *median_spacing]
        median_size = [1, *median_size]

    anisotropy_ratio = max(median_spacing) / max(min(median_spacing), 1e-8)
    is_anisotropic = anisotropy_ratio > 3.0

    class_stats = {}
    for class_id in range(n_classes):
        freq = class_voxel_counts[class_id] / max(total_voxels, 1)
        class_stats[class_names[class_id]] = round(freq, 6)

    image_type = "CT" if any(is_ct_modality) else "MRI"

    fp = DatasetFingerprint(
        dataset_name=dataset_name,
        n_cases=len(items),
        n_classes=n_classes,
        class_names=class_names,
        modalities=modalities,
        spacings=spacings,
        sizes=sizes,
        median_spacing=median_spacing,
        median_size=median_size,
        intensity_stats=intensity_stats,
        class_stats=class_stats,
        anisotropy_ratio=float(anisotropy_ratio),
        is_anisotropic=is_anisotropic,
        image_type=image_type,
        task_type=task_type,
        ignore_class_ids=ignore_class_ids,
        target_class_ids=target_class_ids,
        output_channels=output_channels,
        spatial_dims=spatial_dims if spatial_dims is not None else 3,
    )

    if output_file is not None:
        fp.to_json(output_file)
    return fp
