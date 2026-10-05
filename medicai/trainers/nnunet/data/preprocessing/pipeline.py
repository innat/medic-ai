import concurrent.futures
import hashlib
import json
import multiprocessing
from functools import partial
from pathlib import Path

from keras import ops
import numpy as np
from tqdm import tqdm

from medicai.trainers.nnunet.data.metadata.manifest import DatasetManifest
from medicai.trainers.nnunet.data.preprocessing.normalization import get_normalizer
from medicai.trainers.nnunet.data.preprocessing.resampling import compute_zoom_factors
from medicai.transforms import Resize
from medicai.trainers.nnunet.utils.config import PreprocessedCaseProperties
from medicai.trainers.nnunet.utils.io import (
    collapse_single_channel,
    ensure_spacing,
    get_spatial_shape,
    infer_spatial_dims,
    load_medical_image,
    normalize_layout_and_spacing,
    normalize_layout,
    save_npz,
)


def compute_nonzero_bbox(image: np.ndarray) -> tuple[slice, ...]:
    """Compute a bounding box around the nonzero region of an image.

    Handles channel-last arrays: if ndim > 3, the last axis is treated as
    channels and collapsed via ``any``.  Pure spatial arrays (2D or 3D) are
    used directly.
    """
    if image.ndim == 4:
        # (D, H, W, C) — collapse channels
        mask = np.any(image != 0, axis=-1)
    else:
        # 2D (H, W) or 3D (D, H, W) — purely spatial
        mask = image != 0

    nonzero = np.nonzero(mask)
    if len(nonzero[0]) == 0:
        return tuple(slice(0, s) for s in mask.shape)

    return tuple(
        slice(int(nonzero[ax].min()), int(nonzero[ax].max()) + 1) for ax in range(mask.ndim)
    )


def _bbox_to_list(bbox: tuple[slice, ...]) -> list[list[int]]:
    return [[int(s.start), int(s.stop)] for s in bbox]


def _collect_class_locations(
    label: np.ndarray | None,
    task_type="multi-class",
    target_class_ids=None,
    max_locations_per_class: int = 10_000,
    ignore_class_ids=None,
) -> dict[str, list[list[int]]]:
    if label is None:
        return {}

    target_class_ids = target_class_ids if target_class_ids is not None else []
    ignore_class_ids = ignore_class_ids if ignore_class_ids is not None else []
    class_locations: dict[str, list[list[int]]] = {}

    if task_type in {"multi-label", "region_based"} and label.ndim >= 3:
        n_channels = label.shape[-1]
        class_ids = (
            list(range(1, n_channels + 1))
            if task_type == "region_based"
            else target_class_ids or list(range(1, n_channels + 1))
        )
        for chan_idx in range(n_channels):
            foreground = label[..., chan_idx] > 0
            for ignored_id in ignore_class_ids:
                foreground &= label[..., chan_idx] != ignored_id
            coords = np.argwhere(foreground)
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

    foreground_classes = [
        int(c) for c in np.unique(label) if int(c) > 0 and int(c) not in ignore_class_ids
    ]
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
    label: np.ndarray | None = None,
    pad: int = 0,
) -> tuple[np.ndarray, np.ndarray | None, tuple[slice, ...]]:
    bbox = compute_nonzero_bbox(image)

    if image.ndim == 4:
        shape = image.shape[:3]
    elif image.ndim == 3:
        shape = image.shape
    else:
        shape = image.shape[:2]

    padded_bbox = tuple(
        slice(max(0, s.start - pad), min(sh, s.stop + pad))
        for s, sh in zip(bbox, shape, strict=True)
    )

    if image.ndim == len(shape):
        cropped_image = image[padded_bbox]
    else:
        cropped_image = image[padded_bbox + (slice(None),)]

    if label is None:
        cropped_label = None
    elif label.ndim == len(shape):
        cropped_label = label[padded_bbox]
    else:
        cropped_label = label[padded_bbox + (slice(None),)]

    return cropped_image, cropped_label, padded_bbox


def _resample_channels(images, original_spacing, target_spacing, configuration):
    if len(images[0].shape) != 3 or len(original_spacing) != 3:
        raise ValueError("nnU-Net preprocessing currently expects 3D input volumes and spacing.")
    spatial_shape = images[0].shape
    if configuration == "2d":
        # A 2D network configuration still consumes 3D cases; preserve depth
        # and resample only the in-plane dimensions.
        zoom = compute_zoom_factors(original_spacing[-2:], target_spacing[-2:])
        target_shape = (spatial_shape[0],) + tuple(
            max(1, int(round(size * factor)))
            for size, factor in zip(spatial_shape[-2:], zoom, strict=True)
        )
    else:
        zoom = compute_zoom_factors(original_spacing, target_spacing)
        target_shape = tuple(
            max(1, int(round(size * factor)))
            for size, factor in zip(spatial_shape, zoom, strict=True)
        )

    result = []
    for image in images:
        image_tensor = ops.convert_to_tensor(image[..., np.newaxis], dtype="float32")
        resized = Resize(
            keys=["image"],
            interpolation="trilinear",
            target_shape=target_shape,
            input_layout="DHWC",
        )({"image": image_tensor})
        # TODO: Add cubic interpolation to Resize for closer nnU-Net parity.
        result.append(ops.convert_to_numpy(resized["image"])[..., 0])
    return result


def _resample_label_map(label, original_spacing, target_spacing, configuration):
    if label.ndim != 3 or len(original_spacing) != 3:
        raise ValueError("nnU-Net preprocessing currently expects 3D label volumes and spacing.")
    spatial_shape = label.shape
    if configuration == "2d":
        zoom = compute_zoom_factors(original_spacing[-2:], target_spacing[-2:])
        target_shape = (spatial_shape[0],) + tuple(
            max(1, int(round(size * factor)))
            for size, factor in zip(spatial_shape[-2:], zoom, strict=True)
        )
    else:
        zoom = compute_zoom_factors(original_spacing, target_spacing)
        target_shape = tuple(
            max(1, int(round(size * factor)))
            for size, factor in zip(spatial_shape, zoom, strict=True)
        )

    label_tensor = ops.convert_to_tensor(label[..., np.newaxis], dtype="float32")
    resized = Resize(
        keys=["label"],
        interpolation="nearest",
        target_shape=target_shape,
        input_layout="DHWC",
    )({"label": label_tensor})
    return ops.convert_to_numpy(resized["label"])[..., 0].astype(np.int64)


def _load_image_channels(
    image_paths,
    expected_spatial_dims=None,
    original_spacing_override=None,
    image_layout=None,
):
    images = []
    original_spacing = None
    original_shape = None
    source_affine = None
    source_affine_path = None
    spatial_dims = expected_spatial_dims

    for img_path in image_paths:
        img, affine, _, loaded_spacing = load_medical_image(
            img_path,
        )
        if (
            not np.issubdtype(img.dtype, np.number)
            or np.iscomplexobj(img)
            or not np.all(np.isfinite(img))
        ):
            raise ValueError(f"Image {img_path} must contain finite numeric values.")
        candidate_spacing = (
            original_spacing_override if original_spacing_override is not None else loaded_spacing
        )
        image_dims = infer_spatial_dims(img, spacing=candidate_spacing)
        if spatial_dims is None:
            spatial_dims = image_dims
        img, candidate_spacing = normalize_layout_and_spacing(
            img,
            spatial_dims,
            spacing=candidate_spacing,
            layout=image_layout,
        )
        img = collapse_single_channel(img, spatial_dims)
        image_shape = get_spatial_shape(img, spatial_dims)
        is_nifti = str(img_path).lower().endswith((".nii", ".nii.gz"))

        if original_shape is not None and image_shape != original_shape:
            raise ValueError(
                f"Image modality {img_path} has spatial shape {image_shape}; "
                f"expected {original_shape}; does not match the other modalities."
            )
        if is_nifti:
            affine = np.asarray(affine, dtype=np.float64)
            if source_affine is None:
                source_affine = affine
                source_affine_path = img_path
            elif not np.allclose(affine, source_affine, rtol=1e-5, atol=1e-5):
                raise ValueError(
                    f"NIfTI modality {img_path} has an affine that does not match "
                    f"{source_affine_path}. Register/resample modalities to a common geometry "
                    "first."
                )

        if original_spacing is None:
            original_spacing = ensure_spacing(candidate_spacing, spatial_dims)
            original_shape = image_shape
        else:
            modality_spacing = ensure_spacing(candidate_spacing, spatial_dims)
            if not np.allclose(
                modality_spacing,
                original_spacing,
                rtol=1e-5,
                atol=1e-5,
            ):
                raise ValueError(
                    f"Image modality {img_path} has spacing {modality_spacing}; "
                    f"expected {original_spacing} to match the other modalities."
                )

        if img.ndim > spatial_dims:
            n_channels = img.shape[-1]
            for chan_idx in range(n_channels):
                images.append(img[..., chan_idx])
        else:
            images.append(img.astype(np.float32))

    return images, original_spacing, original_shape, spatial_dims, source_affine


def _validate_label_alignment(
    label_paths,
    image_shape,
    image_affine,
    spatial_dims,
    original_spacing,
    original_spacing_override=None,
    label_layout=None,
):
    """Fail early when a label map is not aligned with its image volume."""
    if label_paths is None:
        return
    label_path = label_paths[0] if isinstance(label_paths, list) else label_paths
    label, affine, _, label_spacing = load_medical_image(label_path)
    expected_label_spacing = original_spacing
    if label_layout is not None:
        target_axes = "HW" if spatial_dims == 2 else "DHW"
        label_axes = label_layout.upper().replace("C", "")
        if len(label_axes) == spatial_dims and set(label_axes) == set(target_axes):
            expected_label_spacing = tuple(
                original_spacing[target_axes.index(axis)] for axis in label_axes
            )
    candidate_spacing = (
        expected_label_spacing
        if original_spacing_override is not None or label_spacing is None
        else label_spacing
    )
    label_dims = infer_spatial_dims(
        label, spacing=candidate_spacing, is_3d=spatial_dims == 3
    )
    label, candidate_spacing = normalize_layout_and_spacing(
        label,
        label_dims,
        spacing=candidate_spacing,
        layout=label_layout,
    )
    label = collapse_single_channel(label, label_dims)
    label_shape = get_spatial_shape(label, label_dims)
    if label_dims != spatial_dims or label_shape != image_shape:
        raise ValueError(
            f"Label {label_path} has spatial shape {label_shape} ({label_dims}D); "
            f"expected {image_shape} ({spatial_dims}D) to match its image."
        )

    is_nifti = str(label_path).lower().endswith((".nii", ".nii.gz"))
    if image_affine is not None:
        if not is_nifti or not np.allclose(
            np.asarray(affine, dtype=np.float64), image_affine, rtol=1e-5, atol=1e-5
        ):
            raise ValueError(
                f"Label {label_path} does not share the image NIfTI affine. "
                "Register/resample the label to the image geometry first."
            )
    if candidate_spacing is not None and not np.allclose(
        ensure_spacing(candidate_spacing, spatial_dims),
        original_spacing,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError(
            f"Label {label_path} spacing {candidate_spacing} does not match "
            f"image spacing {original_spacing}."
        )


def _load_labels(
    label_paths,
    spatial_dims,
    task_type,
    ignore_class_ids,
    target_class_ids,
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
                layout=label_layout,
            )
            label_data = collapse_single_channel(
                label_data,
                label_dims,
            )
            if label_data.ndim > label_dims:
                label_data = label_data[..., 0]
            region_channels = []
            for class_group in regions or []:
                region_channels.append(
                    np.isin(label_data.astype(np.int64), class_group).astype(np.int64)
                )
            return np.stack(region_channels, axis=-1)

        if len(label_list) == 1:
            label_data, _, _, label_spacing = load_medical_image(
                label_list[0],
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
                layout=label_layout,
            )
            label_data = collapse_single_channel(
                label_data,
                label_dims,
            )
            if label_data.ndim > label_dims:
                label_data = label_data[..., 0]
            class_ids = (
                target_class_ids if target_class_ids else list(range(1, int(label_data.max()) + 1))
            )
            label_channels = [(label_data == class_id).astype(np.int64) for class_id in class_ids]
            return np.stack(label_channels, axis=-1)

        label_channels = []
        for label_path in label_list:
            label_data, _, _, label_spacing = load_medical_image(
                label_path,
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
                layout=label_layout,
            )
            label_data = collapse_single_channel(
                label_data,
                label_dims,
            )
            if label_data.ndim > label_dims:
                label_data = label_data[..., 0]
            label_channels.append((label_data > 0).astype(np.int64))
        return np.stack(label_channels, axis=-1)

    label_path = label_paths[0] if isinstance(label_paths, list) else label_paths
    label_data, _, _, label_spacing = load_medical_image(
        label_path,
    )
    candidate_spacing = (
        original_spacing_override if original_spacing_override is not None else label_spacing
    )
    label_dims = infer_spatial_dims(label_data, spacing=candidate_spacing)
    label_dims = max(label_dims, spatial_dims)
    label_data = normalize_layout(
        label_data,
        label_dims,
        layout=label_layout,
    )
    label_data = collapse_single_channel(label_data, label_dims)

    if label_data.ndim > label_dims:
        label_data = label_data[..., 0]

    label_data = label_data.astype(np.int64)
    if task_type == "region_based":
        region_definitions = (
            list(regions.values()) if isinstance(regions, dict) else list(regions or [])
        )
        if not region_definitions:
            raise ValueError("Region-based preprocessing requires at least one region definition.")
        region_targets = np.stack(
            [np.isin(label_data, region_ids).astype(np.int64) for region_ids in region_definitions],
            axis=-1,
        )
        for ignored_id in ignore_class_ids:
            region_targets[label_data == ignored_id, :] = ignored_id
        return region_targets

    if task_type == "binary":
        positive_ids = target_class_ids if target_class_ids else [1]
        binary_mask = np.zeros_like(label_data, dtype=np.int64)
        for class_id in positive_ids:
            binary_mask[label_data == class_id] = 1
        for ignored_id in ignore_class_ids:
            binary_mask[label_data == ignored_id] = ignored_id
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
    original_spacing_override=None,
    image_layout=None,
    label_layout=None,
    label_output=None,
    regions=None,
    use_mask_for_norm=None,
    source_hashes=None,
    cache_signature=None,
):
    output_path = Path(output_path)
    properties_output_path = (
        Path(properties_output_path)
        if properties_output_path is not None
        else output_path.with_suffix(".json")
    )

    images, original_spacing, original_shape, spatial_dims, source_affine = _load_image_channels(
        image_paths=image_paths,
        original_spacing_override=original_spacing_override,
        image_layout=image_layout,
    )
    if not images:
        raise ValueError("No image channels found for case preprocessing.")

    _validate_label_alignment(
        label_paths=label_paths,
        image_shape=original_shape,
        image_affine=source_affine,
        spatial_dims=spatial_dims,
        original_spacing=original_spacing,
        original_spacing_override=original_spacing_override,
        label_layout=label_layout,
    )

    target_spacing = ensure_spacing(target_spacing, spatial_dims)
    label = _load_labels(
        label_paths=label_paths,
        spatial_dims=spatial_dims,
        task_type=item_type,
        ignore_class_ids=ignore_class_ids,
        target_class_ids=target_class_ids,
        original_spacing_override=original_spacing_override,
        label_layout=label_layout,
        label_output=label_output,
        regions=regions,
    )
    # Official nnU-Net order: crop → normalize → resample
    # "normalization MUST happen before resampling or we get huge problems
    #  with resampled nonzero masks no longer fitting the images perfectly!"

    # 1. Stack channels
    image_stack = np.stack(images, axis=-1)
    shape_before_cropping = get_spatial_shape(image_stack, spatial_dims)

    # 2. Crop to nonzero region
    if do_crop:
        image_stack, label, bbox = crop_to_nonzero(
            image_stack,
            label,
        )
        bbox_list = _bbox_to_list(bbox)
    else:
        bbox_list = [[0, int(s)] for s in shape_before_cropping]

    shape_after_cropping = get_spatial_shape(image_stack, spatial_dims)

    # 3. Normalize (before resampling)
    normalized_channels = []

    # Compute nonzero mask for GAP-1 (if any modality needs it)
    global_mask = None
    if use_mask_for_norm and any(use_mask_for_norm.values()):
        global_mask = np.any(image_stack != 0, axis=-1)

    for mod_idx, mod_name in enumerate(modalities):
        mod_data = image_stack[..., mod_idx]
        scheme = (
            normalization_schemes[mod_idx] if mod_idx < len(normalization_schemes) else "z_score"
        )
        mod_use_mask = use_mask_for_norm.get(mod_name, False) if use_mask_for_norm else False

        normalizer = get_normalizer(
            scheme, stats=intensity_stats.get(mod_name), use_mask_for_norm=mod_use_mask
        )

        # Pass the mask specifically if requested
        kwargs = {}
        if mod_use_mask:
            kwargs["mask"] = global_mask

        normalized_channels.append(normalizer(mod_data, **kwargs))

    # 4. Resample to target spacing
    resampled_images = _resample_channels(
        images=normalized_channels,
        original_spacing=original_spacing,
        target_spacing=target_spacing,
        configuration=configuration,
    )

    if label is not None:
        if item_type in {"multi-label", "region_based"} and label.ndim > spatial_dims:
            n_channels = label.shape[-1]
            label_channels = []
            for chan_idx in range(n_channels):
                label_channels.append(
                    _resample_label_map(
                        label[..., chan_idx],
                        original_spacing,
                        target_spacing,
                        configuration,
                    )
                )
            label = np.stack(label_channels, axis=-1)
        else:
            label = _resample_label_map(
                label,
                original_spacing,
                target_spacing,
                configuration,
            )

    image_stack = np.stack(resampled_images, axis=-1).astype(np.float32)
    shape_after_resampling = get_spatial_shape(image_stack, spatial_dims)
    metadata_target_spacing = list(target_spacing)
    if configuration == "2d" and spatial_dims == 3:
        metadata_target_spacing = [original_spacing[0], *target_spacing[-2:]]

    save_dict = {
        "image": image_stack,
        "spacing": np.asarray(metadata_target_spacing, dtype=np.float32),
    }
    if label is not None:
        save_dict["label"] = label.astype(np.int64)
    save_npz(save_dict, output_path)

    properties = PreprocessedCaseProperties(
        case_id=output_path.stem,
        original_spacing=[float(s) for s in original_spacing],
        target_spacing=[float(s) for s in metadata_target_spacing],
        original_shape=[int(s) for s in original_shape],
        shape_after_resampling=[int(s) for s in shape_after_resampling],
        shape_after_cropping=[int(s) for s in shape_after_cropping],
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
            ignore_class_ids=ignore_class_ids,
        ),
        item_type=item_type,
        spatial_dims=spatial_dims,
        source_affine=source_affine.tolist() if source_affine is not None else None,
        source_layout=image_layout,
        source_spacing=(
            [float(value) for value in original_spacing_override]
            if original_spacing_override is not None
            else None
        ),
        source_hashes=source_hashes,
        cache_signature=cache_signature,
    )
    properties.to_json(properties_output_path)

    return {
        "image": image_stack,
        "label": label,
        "spacing": metadata_target_spacing,
        "properties": properties,
    }


def _hash_source_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _process_item_helper(
    item,
    output_dir,
    properties_dir,
    manifest,
    target_spacing,
    plan,
    fingerprint,
    configuration,
    incremental,
):
    output_path = output_dir / f"{item.case_id}.npz"
    properties_output_path = properties_dir / f"{item.case_id}.json"
    image_layout = item.image_layout or manifest.image_layout
    label_layout = item.label_layout or manifest.label_layout
    label_output = item.label_output or manifest.label_output
    label_paths = item.labels if isinstance(item.labels, list) else [item.labels]
    source_paths = [*item.images, *(path for path in label_paths if path is not None)]
    source_hashes = {str(path): _hash_source_file(path) for path in source_paths}
    signature_data = {
        "cache_version": 1,
        "source_hashes": source_hashes,
        "configuration": configuration,
        "target_spacing": list(target_spacing),
        "normalization_schemes": list(plan.normalization_schemes),
        "intensity_stats": fingerprint.intensity_stats,
        "use_mask_for_norm": fingerprint.use_mask_for_norm,
        "modalities": list(fingerprint.modalities),
        "task_type": item.task_type or manifest.task_type,
        "ignore_class_ids": manifest.ignore_class_ids,
        "target_class_ids": manifest.target_class_ids,
        "label_output": label_output,
        "regions": item.regions or manifest.regions,
        "spacing": item.spacing,
        "image_layout": image_layout,
        "label_layout": label_layout,
    }
    cache_signature = hashlib.sha256(
        json.dumps(signature_data, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()
    if incremental and output_path.exists() and properties_output_path.exists():
        try:
            cached = PreprocessedCaseProperties.from_json(properties_output_path)
            with np.load(output_path, allow_pickle=False) as cache:
                valid_cache = "image" in cache and "spacing" in cache
            if cached.cache_signature == cache_signature and valid_cache:
                return {"case_id": item.case_id, "status": "skipped"}
        except Exception:
            pass
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
        original_spacing_override=item.spacing,
        image_layout=image_layout,
        label_layout=label_layout,
        label_output=label_output,
        regions=item.regions or manifest.regions,
        use_mask_for_norm=fingerprint.use_mask_for_norm,
        source_hashes=source_hashes,
        cache_signature=cache_signature,
    )
    return {"case_id": item.case_id, "status": "processed"}


def preprocess_dataset(
    fingerprint,
    plan,
    output_dir,
    configuration="3d_fullres",
    max_cases=None,
    manifest_file=None,
    num_workers=None,
    incremental=True,
):

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

    target_spacing = net_cfg.target_spacing or plan.target_spacing

    if manifest_file is None or not Path(manifest_file).exists():
        raise ValueError(
            "manifest_file must be provided for preprocessing. "
            "Legacy MSD layout is no longer supported."
        )

    manifest = DatasetManifest.from_json(manifest_file)
    if max_cases is not None and max_cases < 0:
        raise ValueError("max_cases must be non-negative or None.")
    if num_workers is not None and num_workers < 1:
        raise ValueError("num_workers must be a positive integer or None.")
    items = manifest.items if max_cases is None else manifest.items[:max_cases]

    worker_fn = partial(
        _process_item_helper,
        output_dir=output_dir,
        properties_dir=properties_dir,
        manifest=manifest,
        target_spacing=target_spacing,
        plan=plan,
        fingerprint=fingerprint,
        configuration=configuration,
        incremental=incremental,
    )

    workers = num_workers if num_workers is not None else max(1, multiprocessing.cpu_count() - 1)
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(
                tqdm(
                    executor.map(worker_fn, items),
                    total=len(items),
                    desc=f"Preprocessing ({configuration})",
                )
            )
    else:
        results = [
            worker_fn(item) for item in tqdm(items, desc=f"Preprocessing ({configuration})")
        ]

    return {
        "configuration": configuration,
        "output_dir": str(output_dir),
        "processed_cases": [
            result["case_id"] for result in results if result["status"] == "processed"
        ],
        "skipped_cases": [
            result["case_id"] for result in results if result["status"] == "skipped"
        ],
    }
