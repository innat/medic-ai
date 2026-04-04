from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from medicai.trainer.nnunet.data.manifest import DatasetManifest
from medicai.trainer.nnunet.data.normalization import get_normalizer
from medicai.trainer.nnunet.data.resampling import (
    resample_image,
    resample_image_2d,
    resample_label,
)
from medicai.trainer.nnunet.utils.config import PreprocessedCaseProperties
from medicai.trainer.nnunet.utils.io import (
    collapse_single_channel,
    ensure_spacing,
    get_spatial_shape,
    infer_spatial_dims,
    load_medical_image,
    normalize_layout,
    save_npz,
)


def compute_nonzero_bbox(image: np.ndarray, ensure_channel_last=True) -> Tuple[slice, ...]:
    if image.ndim == 4:
        mask = np.any(image != 0, axis=-1 if ensure_channel_last else 0)
    elif image.ndim == 3 and ensure_channel_last and image.shape[-1] <= 4:
        mask = np.any(image != 0, axis=-1)
    else:
        mask = image != 0

    nonzero = np.nonzero(mask)
    if len(nonzero[0]) == 0:
        return tuple(slice(0, s) for s in mask.shape)

    return tuple(
        slice(int(nonzero[ax].min()), int(nonzero[ax].max()) + 1) for ax in range(mask.ndim)
    )


def _bbox_to_list(bbox: Tuple[slice, ...]) -> List[List[int]]:
    return [[int(s.start), int(s.stop)] for s in bbox]


def _collect_class_locations(
    label: Optional[np.ndarray],
    task_type="multi-class",
    target_class_ids=None,
    max_locations_per_class: int = 10_000,
) -> Dict[str, List[List[int]]]:
    if label is None:
        return {}

    target_class_ids = target_class_ids if target_class_ids is not None else []
    class_locations: Dict[str, List[List[int]]] = {}

    if task_type == "multi-label" and label.ndim >= 3:
        n_channels = label.shape[-1]
        class_ids = target_class_ids if target_class_ids else list(range(1, n_channels + 1))
        for chan_idx in range(n_channels):
            coords = np.argwhere(label[..., chan_idx] > 0)
            if coords.size == 0:
                continue
            if len(coords) > max_locations_per_class:
                indices = np.linspace(
                    0,
                    len(coords) - 1,
                    num=max_locations_per_class,
                    dtype=np.int64,
                )
                coords = coords[indices]
            class_locations[str(class_ids[chan_idx])] = coords.astype(np.int64).tolist()
        return class_locations

    foreground_classes = [int(c) for c in np.unique(label) if int(c) > 0]
    for class_id in foreground_classes:
        coords = np.argwhere(label == class_id)
        if coords.size == 0:
            continue
        if len(coords) > max_locations_per_class:
            indices = np.linspace(
                0,
                len(coords) - 1,
                num=max_locations_per_class,
                dtype=np.int64,
            )
            coords = coords[indices]
        class_locations[str(class_id)] = coords.astype(np.int64).tolist()
    return class_locations


def crop_to_nonzero(
    image: np.ndarray,
    label: Optional[np.ndarray] = None,
    pad: int = 0,
    ensure_channel_last=True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[slice, ...]]:
    bbox = compute_nonzero_bbox(image, ensure_channel_last=ensure_channel_last)

    if image.ndim == 4:
        shape = image.shape[:3] if ensure_channel_last else image.shape[1:]
    else:
        shape = image.shape[:2]

    padded_bbox = tuple(
        slice(max(0, s.start - pad), min(sh, s.stop + pad)) for s, sh in zip(bbox, shape)
    )

    if image.ndim == len(shape):
        cropped_image = image[padded_bbox]
    elif ensure_channel_last:
        cropped_image = image[padded_bbox + (slice(None),)]
    else:
        cropped_image = image[(slice(None),) + padded_bbox]

    if label is None:
        cropped_label = None
    elif label.ndim == len(shape):
        cropped_label = label[padded_bbox]
    elif ensure_channel_last:
        cropped_label = label[padded_bbox + (slice(None),)]
    else:
        cropped_label = label[(slice(None),) + padded_bbox]

    return cropped_image, cropped_label, padded_bbox


def _resample_channels(images, original_spacing, target_spacing, configuration):
    if configuration == "2d":
        image_spacing_2d = original_spacing[-2:]
        target_spacing_2d = target_spacing[-2:]
        return [
            resample_image_2d(img, image_spacing_2d, target_spacing_2d, order=3) for img in images
        ]
    return [resample_image(img, original_spacing, target_spacing, order=3) for img in images]


def _resample_label_map(label, original_spacing, target_spacing, configuration):
    if configuration == "2d":
        return resample_image_2d(
            label.astype(np.float32),
            original_spacing[-2:],
            target_spacing[-2:],
            order=0,
        ).astype(np.int64)
    return resample_label(label.astype(np.float32), original_spacing, target_spacing).astype(
        np.int64
    )


def _load_image_channels(
    image_paths,
    ensure_channel_last,
    expected_spatial_dims=None,
    original_spacing_override=None,
    image_layout=None,
):
    images = []
    original_spacing = None
    original_shape = None
    spatial_dims = expected_spatial_dims

    for img_path in image_paths:
        img, _, _, loaded_spacing = load_medical_image(
            img_path,
            ensure_channel_last=ensure_channel_last,
        )
        candidate_spacing = (
            original_spacing_override if original_spacing_override is not None else loaded_spacing
        )
        image_dims = infer_spatial_dims(img, spacing=candidate_spacing)
        if spatial_dims is None:
            spatial_dims = image_dims
        img = normalize_layout(
            img,
            spatial_dims,
            ensure_channel_last=ensure_channel_last,
            layout=image_layout,
        )
        img = collapse_single_channel(img, spatial_dims, ensure_channel_last=ensure_channel_last)

        if original_spacing is None:
            original_spacing = ensure_spacing(candidate_spacing, spatial_dims)
            original_shape = get_spatial_shape(
                img, spatial_dims, ensure_channel_last=ensure_channel_last
            )

        if img.ndim > spatial_dims:
            channel_index = -1 if ensure_channel_last else 0
            n_channels = img.shape[channel_index]
            for chan_idx in range(n_channels):
                images.append(img[..., chan_idx] if ensure_channel_last else img[chan_idx])
        else:
            images.append(img.astype(np.float32))

    return images, original_spacing, original_shape, spatial_dims


def _load_labels(
    label_paths,
    spatial_dims,
    task_type,
    ignore_class_ids,
    target_class_ids,
    ensure_channel_last,
    original_spacing_override=None,
    label_layout=None,
    label_output="auto",
    regions=None,
):
    if label_paths is None:
        return None

    ignore_class_ids = ignore_class_ids if ignore_class_ids is not None else []

    if task_type == "multi-label":
        label_list = label_paths if isinstance(label_paths, list) else [label_paths]
        if label_output == "regions":
            label_data, _, _, label_spacing = load_medical_image(
                label_list[0],
                ensure_channel_last=ensure_channel_last,
            )
            candidate_spacing = (
                original_spacing_override
                if original_spacing_override is not None
                else label_spacing
            )
            label_dims = infer_spatial_dims(label_data, spacing=candidate_spacing)
            label_dims = max(label_dims, spatial_dims)
            label_data = normalize_layout(
                label_data,
                label_dims,
                ensure_channel_last=ensure_channel_last,
                layout=label_layout,
            )
            label_data = collapse_single_channel(
                label_data,
                label_dims,
                ensure_channel_last=ensure_channel_last,
            )
            if label_data.ndim > label_dims:
                label_data = label_data[..., 0] if ensure_channel_last else label_data[0]
            region_channels = []
            for class_group in regions or []:
                region_channels.append(
                    np.isin(label_data.astype(np.int64), class_group).astype(np.int64)
                )
            return np.stack(region_channels, axis=-1 if ensure_channel_last else 0)

        if len(label_list) == 1:
            label_data, _, _, label_spacing = load_medical_image(
                label_list[0],
                ensure_channel_last=ensure_channel_last,
            )
            candidate_spacing = (
                original_spacing_override
                if original_spacing_override is not None
                else label_spacing
            )
            label_dims = infer_spatial_dims(label_data, spacing=candidate_spacing)
            label_dims = max(label_dims, spatial_dims)
            label_data = normalize_layout(
                label_data,
                label_dims,
                ensure_channel_last=ensure_channel_last,
                layout=label_layout,
            )
            label_data = collapse_single_channel(
                label_data,
                label_dims,
                ensure_channel_last=ensure_channel_last,
            )
            if label_data.ndim > label_dims:
                label_data = label_data[..., 0] if ensure_channel_last else label_data[0]
            class_ids = (
                target_class_ids if target_class_ids else list(range(1, int(label_data.max()) + 1))
            )
            label_channels = [(label_data == class_id).astype(np.int64) for class_id in class_ids]
            return np.stack(label_channels, axis=-1 if ensure_channel_last else 0)

        label_channels = []
        for label_path in label_list:
            label_data, _, _, label_spacing = load_medical_image(
                label_path,
                ensure_channel_last=ensure_channel_last,
            )
            candidate_spacing = (
                original_spacing_override
                if original_spacing_override is not None
                else label_spacing
            )
            label_dims = infer_spatial_dims(label_data, spacing=candidate_spacing)
            label_dims = max(label_dims, spatial_dims)
            label_data = normalize_layout(
                label_data,
                label_dims,
                ensure_channel_last=ensure_channel_last,
                layout=label_layout,
            )
            label_data = collapse_single_channel(
                label_data,
                label_dims,
                ensure_channel_last=ensure_channel_last,
            )
            if label_data.ndim > label_dims:
                label_data = label_data[..., 0] if ensure_channel_last else label_data[0]
            label_channels.append((label_data > 0).astype(np.int64))
        return np.stack(label_channels, axis=-1 if ensure_channel_last else 0)

    label_path = label_paths[0] if isinstance(label_paths, list) else label_paths
    label_data, _, _, label_spacing = load_medical_image(
        label_path,
        ensure_channel_last=ensure_channel_last,
    )
    candidate_spacing = (
        original_spacing_override if original_spacing_override is not None else label_spacing
    )
    label_dims = infer_spatial_dims(label_data, spacing=candidate_spacing)
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

    if label_data.ndim > label_dims:
        label_data = label_data[..., 0] if ensure_channel_last else label_data[0]

    label_data = label_data.astype(np.int64)
    for ignored_id in ignore_class_ids:
        label_data[label_data == ignored_id] = -1

    if task_type == "binary":
        positive_ids = target_class_ids if target_class_ids else [1]
        binary_mask = np.zeros_like(label_data, dtype=np.int64)
        for class_id in positive_ids:
            binary_mask[label_data == class_id] = 1
        binary_mask[label_data < 0] = -1
        return binary_mask

    return label_data


def preprocess_case(
    image_paths,
    label_paths,
    target_spacing,
    normalization_schemes,
    intensity_stats,
    modalities,
    output_path,
    configuration="3d_fullres",
    do_crop=True,
    properties_output_path=None,
    ignore_class_ids=None,
    target_class_ids=None,
    item_type="multi-class",
    ensure_channel_last=True,
    original_spacing_override=None,
    image_layout=None,
    label_layout=None,
    label_output="auto",
    regions=None,
):
    output_path = Path(output_path)
    properties_output_path = (
        Path(properties_output_path)
        if properties_output_path is not None
        else output_path.with_suffix(".json")
    )

    images, original_spacing, original_shape, spatial_dims = _load_image_channels(
        image_paths=image_paths,
        ensure_channel_last=ensure_channel_last,
        original_spacing_override=original_spacing_override,
        image_layout=image_layout,
    )
    if not images:
        raise ValueError("No image channels found for case preprocessing.")

    target_spacing = ensure_spacing(target_spacing, spatial_dims)
    label = _load_labels(
        label_paths=label_paths,
        spatial_dims=spatial_dims,
        task_type=item_type,
        ignore_class_ids=ignore_class_ids,
        target_class_ids=target_class_ids,
        ensure_channel_last=ensure_channel_last,
        original_spacing_override=original_spacing_override,
        label_layout=label_layout,
        label_output=label_output,
        regions=regions,
    )

    resampled_images = _resample_channels(
        images=images,
        original_spacing=original_spacing,
        target_spacing=target_spacing,
        configuration=configuration,
    )

    if label is not None:
        if item_type == "multi-label" and label.ndim > spatial_dims:
            n_channels = label.shape[-1] if ensure_channel_last else label.shape[0]
            label_channels = []
            for chan_idx in range(n_channels):
                label_channels.append(
                    _resample_label_map(
                        label[..., chan_idx] if ensure_channel_last else label[chan_idx],
                        original_spacing,
                        target_spacing,
                        configuration,
                    )
                )
            label = np.stack(label_channels, axis=-1 if ensure_channel_last else 0)
        else:
            label = _resample_label_map(
                label,
                original_spacing,
                target_spacing,
                configuration,
            )

    image_stack = np.stack(resampled_images, axis=-1 if ensure_channel_last else 0)
    shape_after_resampling = get_spatial_shape(
        image_stack,
        spatial_dims,
        ensure_channel_last=ensure_channel_last,
    )

    if do_crop and configuration != "2d":
        image_stack, label, bbox = crop_to_nonzero(
            image_stack,
            label,
            ensure_channel_last=ensure_channel_last,
        )
        bbox_list = _bbox_to_list(bbox)
    else:
        bbox_list = [[0, int(s)] for s in shape_after_resampling]

    normalized_channels = []
    for mod_idx, mod_name in enumerate(modalities):
        if ensure_channel_last:
            mod_data = image_stack[..., mod_idx]
        else:
            mod_data = image_stack[mod_idx]
        scheme = (
            normalization_schemes[mod_idx] if mod_idx < len(normalization_schemes) else "z_score"
        )
        normalizer = get_normalizer(scheme, stats=intensity_stats.get(mod_name))
        normalized_channels.append(normalizer(mod_data))

    image_stack = np.stack(
        normalized_channels,
        axis=-1 if ensure_channel_last else 0,
    ).astype(np.float32)

    save_dict = {"data": image_stack, "spacing": np.asarray(target_spacing, dtype=np.float32)}
    if label is not None:
        save_dict["seg"] = label.astype(np.int64)
    save_npz(save_dict, output_path)

    properties = PreprocessedCaseProperties(
        case_id=output_path.stem,
        original_spacing=[float(s) for s in original_spacing],
        target_spacing=[float(s) for s in target_spacing],
        original_shape=[int(s) for s in original_shape],
        shape_after_resampling=[int(s) for s in shape_after_resampling],
        shape_after_cropping=get_spatial_shape(
            image_stack,
            spatial_dims,
            ensure_channel_last=ensure_channel_last,
        ),
        bbox=bbox_list,
        modalities=list(modalities),
        normalization_schemes=list(normalization_schemes),
        image_files=[str(Path(p)) for p in image_paths],
        label_file=(
            None
            if label_paths is None
            else (
                [str(Path(p)) for p in label_paths]
                if isinstance(label_paths, list)
                else str(label_paths)
            )
        ),
        class_locations=_collect_class_locations(
            label,
            task_type=item_type,
            target_class_ids=target_class_ids,
        ),
        item_type=item_type,
        spatial_dims=spatial_dims,
    )
    properties.to_json(properties_output_path)

    return {
        "data": image_stack,
        "seg": label,
        "spacing": target_spacing,
        "properties": properties,
    }


def preprocess_dataset(
    fingerprint,
    plan,
    output_dir,
    configuration="3d_fullres",
    num_workers=1,
    max_cases=None,
    manifest_file=None,
    ensure_channel_last=True,
):
    del num_workers

    output_dir = Path(output_dir) / configuration
    properties_dir = output_dir / plan.preprocessed_properties_dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    properties_dir.mkdir(parents=True, exist_ok=True)

    net_cfg_map = {
        "3d_fullres": plan.plan_3d_fullres,
        "3d_lowres": plan.plan_3d_lowres,
        "2d": plan.plan_2d,
    }
    net_cfg = net_cfg_map.get(configuration)
    if net_cfg is None:
        raise ValueError(
            f"Plan does not contain configuration '{configuration}'. "
            f"Available: {[k for k, v in net_cfg_map.items() if v is not None]}"
        )

    if configuration == "2d" and getattr(fingerprint, "spatial_dims", 3) == 3:
        target_spacing = plan.target_spacing[1:]
    else:
        target_spacing = plan.target_spacing

    if manifest_file is None or not Path(manifest_file).exists():
        raise ValueError(
            "manifest_file must be provided for preprocessing. "
            "Legacy MSD layout is no longer supported."
        )

    manifest = DatasetManifest.from_json(manifest_file)
    items = manifest.items[:max_cases] if max_cases is not None else manifest.items

    for item in tqdm(items, desc=f"Preprocessing ({configuration})"):
        output_path = output_dir / f"{item.case_id}.npz"
        properties_output_path = properties_dir / f"{item.case_id}.json"

        if output_path.exists() and properties_output_path.exists():
            continue

        image_layout = item.image_layout or manifest.image_layout
        label_layout = item.label_layout or manifest.label_layout
        label_output = item.label_output or manifest.label_output
        preprocess_case(
            image_paths=[Path(p) for p in item.images],
            label_paths=item.labels,
            target_spacing=target_spacing,
            normalization_schemes=plan.normalization_schemes,
            intensity_stats=fingerprint.intensity_stats,
            modalities=fingerprint.modalities,
            output_path=output_path,
            configuration=configuration,
            do_crop=True,
            properties_output_path=properties_output_path,
            ignore_class_ids=manifest.ignore_class_ids,
            target_class_ids=manifest.target_class_ids,
            item_type=item.task_type or manifest.task_type,
            ensure_channel_last=ensure_channel_last,
            original_spacing_override=item.spacing,
            image_layout=image_layout,
            label_layout=label_layout,
            label_output=label_output,
            regions=item.regions or manifest.regions,
        )
