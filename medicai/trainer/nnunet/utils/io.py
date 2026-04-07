from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from skimage import io as skio


def load_nifti(path):
    """
    Load a NIfTI file and normalize axes to channel-last order.
    """

    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)

    if data.ndim == 2:
        zooms = img.header.get_zooms()
        spacing = tuple(float(z) for z in reversed(zooms[:2]))
        return data, img.affine, img.header, spacing
    if data.ndim == 3:
        data = data.transpose(2, 1, 0)
    elif data.ndim == 4:
        data = data.transpose(2, 1, 0, 3)
    else:
        raise ValueError(f"Unexpected NIfTI ndim={data.ndim} in {path}")

    zooms = img.header.get_zooms()
    spatial_dims = 3 if data.ndim in (3, 4) else 2
    spacing = tuple(float(z) for z in reversed(zooms[:spatial_dims]))
    return data, img.affine, img.header, spacing


def save_nifti(
    image,
    affine,
    path,
    header=None,
    dtype=np.int16,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.ndim == 2:
        data_out = image.astype(dtype)
    elif image.ndim == 3:
        data_out = image.transpose(2, 1, 0).astype(dtype)
    elif image.ndim == 4:
        data_out = image.transpose(3, 2, 1, 0).astype(dtype)
    else:
        raise ValueError(f"Unexpected ndim={image.ndim}")

    nii = nib.Nifti1Image(data_out, affine=affine, header=header)
    nib.save(nii, str(path))


def save_npz(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **data)


def load_npz(path):
    archive = np.load(str(path), allow_pickle=False)
    return dict(archive)


def list_medical_images(directory):
    directory = Path(directory)
    extensions = (
        "*.nii.gz",
        "*.nii",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.tif",
        "*.tiff",
        "*.dcm",
    )
    files = []
    for ext in extensions:
        files.extend(directory.rglob(ext))
    return sorted(files)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_case_id(path):
    """
    Extract case identifier from a file path by removing the extension.

    Handles: .nii.gz, .nii, .npz, .png, .jpg, .jpeg, .tif, .tiff, .dcm

    Note: Unlike ``normalize_case_id()`` in cross_validation.py, this does NOT
    strip modality suffixes (e.g., ``_0000``). Use this for file identification;
    use ``normalize_case_id`` for grouping multi-modal files by case.
    """
    name = Path(path).name
    for suffix in (
        ".nii.gz",
        ".nii",
        ".npz",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".dcm",
    ):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def infer_spatial_dims(data, spacing=None, is_3d=None):
    """
    Infer the number of spatial dimensions (2 or 3) from the data shape and spacing.

    Warning: The heuristic data.shape[0] <= 4 may misclassify 3D volumes with very
    small depth (e.g., shape (3, 256, 256)) as 2D with channels. Consider providing
    is_3d=True explicitly to avoid this issue for ambiguous shapes.
    """
    if is_3d is not None:
        return 3 if is_3d else 2
    if spacing is not None and len(spacing) >= 3 and data.ndim >= 3:
        return 3
    if data.ndim <= 2:
        return 2
    if data.ndim >= 4:
        return 3
    if data.ndim == 3:
        if data.shape[0] <= 4 or data.shape[-1] <= 4:
            return 2
        return 3
    return 2


def ensure_spacing(spacing, spatial_dims):
    if spacing is None:
        return [1.0] * spatial_dims

    spacing = [float(v) for v in spacing]
    if len(spacing) == spatial_dims:
        return spacing
    if len(spacing) > spatial_dims:
        return spacing[-spatial_dims:]
    if len(spacing) < spatial_dims:
        return [1.0] * (spatial_dims - len(spacing)) + spacing
    return spacing


def get_spatial_shape(data, spatial_dims):
    if data.ndim == spatial_dims:
        return list(data.shape)
    return list(data.shape[:spatial_dims])


def _normalize_layout_name(layout):
    if layout is None:
        return None
    return "".join(ch for ch in str(layout).upper() if ch.isalpha())


def _target_layout(spatial_dims):
    spatial_axes = "HW" if spatial_dims == 2 else "DHW"
    return spatial_axes + "C"


def _transpose_to_layout(data, source_layout, target_layout):
    axis_map = {axis: idx for idx, axis in enumerate(source_layout)}
    return np.transpose(data, [axis_map[axis] for axis in target_layout])


def normalize_layout(data, spatial_dims, layout=None):
    """
    Normalize arrays to channel-last format using specific layout hints.
    """

    layout = _normalize_layout_name(layout)
    if layout is not None and len(layout) == data.ndim:
        spatial_target = "HW" if spatial_dims == 2 else "DHW"
        target = spatial_target if data.ndim == spatial_dims else _target_layout(spatial_dims)
        if layout != target:
            return _transpose_to_layout(data, layout, target)
        return data

    if data.ndim == spatial_dims:
        return data
    if data.ndim != spatial_dims + 1:
        return data

    if data.shape[-1] <= 4:
        return data
    if data.shape[0] <= 4:
        return np.moveaxis(data, 0, -1)
    return data


def collapse_single_channel(data, spatial_dims):
    if data.ndim == spatial_dims + 1:
        if data.shape[-1] == 1:
            return np.squeeze(data, axis=-1)
    return data


def load_medical_image(path):
    """
    Load an image (NIfTI, DICOM, or common raster formats).
    Returns (data, affine, header, spacing).
    """

    path_str = str(path).lower()
    if path_str.endswith(".nii.gz") or path_str.endswith(".nii"):
        return load_nifti(path)

    if path_str.endswith(".dcm"):
        try:
            import pydicom
        except ImportError as exc:
            raise ImportError(
                "pydicom is required to read DICOM files. Install with 'pip install pydicom'."
            ) from exc

        dcm = pydicom.dcmread(str(path))
        data = dcm.pixel_array.astype(np.float32)
        spacing = None
        if hasattr(dcm, "PixelSpacing"):
            spacing = tuple(float(s) for s in dcm.PixelSpacing)
        if hasattr(dcm, "SliceThickness") and data.ndim >= 3:
            base = list(spacing) if spacing is not None else [1.0, 1.0]
            spacing = (float(dcm.SliceThickness), *base)
        return data, np.eye(4), dcm, spacing

    if path_str.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        data = skio.imread(str(path)).astype(np.float32)
        return data, np.eye(4), None, None

    raise ValueError(f"Unsupported file format: {path}")


def save_medical_image(
    image,
    affine,
    path,
    header=None,
    dtype=np.int16,
):
    path_str = str(path).lower()
    if path_str.endswith(".nii.gz") or path_str.endswith(".nii"):
        save_nifti(image, affine, path, header, dtype)
        return
    if path_str.endswith(".dcm"):
        raise NotImplementedError(
            "Saving to DICOM is not supported. Use .nii.gz or another structured format."
        )
    if path_str.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image_out = image.astype(dtype)
        if image_out.ndim == 3 and image_out.shape[0] <= 4:
            image_out = image_out.transpose(1, 2, 0)
        if dtype in (np.int16, np.int32, np.int64) and path_str.endswith((".png", ".jpg", ".jpeg")):
            image_out = image_out.astype(np.uint8)
        if image_out.ndim == 3 and image_out.shape[-1] == 1:
            image_out = np.squeeze(image_out, axis=-1)
        skio.imsave(str(path), image_out, check_contrast=False)
        return
    raise ValueError(f"Unsupported save format: {path}")
