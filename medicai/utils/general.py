import inspect
import re
from typing import Any, List, Sequence, Tuple

import numpy as np
from keras import ops


def hide_warnings():
    import logging
    import os
    import sys
    import warnings

    # Disable Python warnings
    def warn(*args, **kwargs):
        pass

    warnings.warn = warn
    warnings.simplefilter(action="ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Disable Python logging
    logging.disable(logging.WARNING)
    logging.getLogger("tensorflow").disabled = True

    # TensorFlow environment variables
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"

    # Disable TensorFlow debugging information
    if "tensorflow" in sys.modules:
        tf = sys.modules["tensorflow"]
        tf.get_logger().setLevel("ERROR")
        tf.autograph.set_verbosity(0)

    # Disable ABSL (Ten1orFlow dependency) logging
    try:
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
        # Redirect ABSL logs to null
        absl.logging.get_absl_handler().python_handler.stream = open(os.devnull, "w")
    except (ImportError, AttributeError):
        pass


def ensure_tuple_rep(val: Any, rep: int) -> Tuple[Any, ...]:
    """Ensure `val` is a tuple of length `rep`."""
    if isinstance(val, (int, float)):
        return (val,) * rep
    if len(val) == rep:
        return tuple(val)
    raise ValueError(f"Length of `val` ({len(val)}) must match `rep` ({rep}).")


def fall_back_tuple(val: Any, fallback: Sequence[int]) -> Tuple[int, ...]:
    """Ensure `val` is a tuple of the same length as `fallback`."""
    if val is None:
        return tuple(fallback)
    if isinstance(val, int):
        return (val,) * len(fallback)
    if len(val) != len(fallback):
        raise ValueError(f"Length of `val` ({len(val)}) must match `fallback` ({len(fallback)}).")
    return tuple(val)


def get_scan_interval(
    image_size: Sequence[int],
    roi_size: Sequence[int],
    num_spatial_dims: int,
    overlap: Sequence[float],
) -> Tuple[int, ...]:
    """Compute scan intervals based on image size, roi size, and overlap."""
    scan_interval = []
    for i, o in zip(range(num_spatial_dims), overlap):
        if roi_size[i] == image_size[i]:
            scan_interval.append(roi_size[i])
        else:
            interval = int(roi_size[i] * (1 - o))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def dense_patch_slices(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    scan_interval: Sequence[int],
    return_slice: bool = True,
) -> List[Tuple[slice, ...]]:
    num_spatial_dims = len(image_size)

    # Calculate the number of patches along each dimension
    scan_num = []
    for i in range(num_spatial_dims):
        if scan_interval[i] == 0:
            scan_num.append(1)
        else:
            num = (image_size[i] + scan_interval[i] - 1) // scan_interval[
                i
            ]  # Equivalent to math.ceil
            scan_dim = next(
                (d for d in range(num) if d * scan_interval[i] + patch_size[i] >= image_size[i]),
                None,
            )
            scan_num.append(scan_dim + 1 if scan_dim is not None else 1)

    # Generate start indices for each dimension
    starts = []
    for dim in range(num_spatial_dims):
        dim_starts = []
        for idx in range(scan_num[dim]):
            start_idx = idx * scan_interval[dim]
            start_idx -= max(start_idx + patch_size[dim] - image_size[dim], 0)
            dim_starts.append(start_idx)
        starts.append(dim_starts)

    # Generate all combinations of start indices
    out = []
    from itertools import product

    for start_indices in product(*starts):
        if return_slice:
            out.append(
                tuple(slice(start, start + patch_size[d]) for d, start in enumerate(start_indices))
            )
        else:
            out.append(
                tuple((start, start + patch_size[d]) for d, start in enumerate(start_indices))
            )

    return out


def compute_importance_map(
    patch_size: Sequence[int],
    mode: str = "constant",
    sigma_scale: Sequence[float] = (0.125,),
    dtype=np.float32,
) -> np.ndarray:
    """Compute importance map for blending."""
    if mode == "constant":
        return np.ones(patch_size, dtype=dtype)

    elif mode == "gaussian":
        sigma = [s * p for s, p in zip(sigma_scale, patch_size)]
        grid = np.meshgrid(*[np.arange(p, dtype=dtype) for p in patch_size], indexing="ij")
        center = [(p - 1) / 2 for p in patch_size]
        dist = np.sqrt(sum((g - c) ** 2 for g, c in zip(grid, center)))
        return np.exp(-0.5 * (dist / sigma) ** 2)

    else:
        raise ValueError(f"Unsupported mode: {mode}")


def get_valid_patch_size(image_size: Sequence[int], patch_size: Sequence[int]) -> Tuple[int, ...]:
    """
    Ensure the patch size is valid (i.e., patch_size <= image_size).
    """
    return tuple(min(p, i) for p, i in zip(patch_size, image_size))


def crop_output(
    output: np.ndarray, pad_size: Sequence[Sequence[int]], original_size: Sequence[int]
) -> np.ndarray:
    """
    Crop the output to remove padding.

    Args:
        output: Output array with shape (batch_size, *padded_size, channels).
        pad_size: Padding applied to the input tensor.
        original_size: Original spatial size of the input tensor.

    Returns:
        Cropped output array with shape (batch_size, *original_size, channels).
    """
    crop_slices = [slice(None)]  # Keep batch dimension
    for i in range(len(original_size)):
        start = pad_size[i + 1][0]  # Skip batch dimension
        end = start + original_size[i]
        crop_slices.append(slice(start, end))
    crop_slices.append(slice(None))  # Keep channel dimension

    # Convert the list of slices to a tuple for proper indexing
    return output[tuple(crop_slices)]


def resize_volumes(volumes, depth, height, width, method="trilinear", align_corners=False):
    def trilinear_resize(volumes, depth, height, width, align_corners):
        original_dtype = volumes.dtype
        volumes = ops.cast(volumes, "float32")
        batch_size, in_d, in_h, in_w, channels = ops.shape(volumes)

        if align_corners:
            # Map corner to corner
            z_coords = ops.linspace(0.0, ops.cast(in_d - 1, "float32"), depth)
            y_coords = ops.linspace(0.0, ops.cast(in_h - 1, "float32"), height)
            x_coords = ops.linspace(0.0, ops.cast(in_w - 1, "float32"), width)
        else:
            # More accurate PyTorch-compatible mapping
            # Ref: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            scale_d = ops.cast(in_d, "float32") / ops.cast(depth, "float32")
            scale_h = ops.cast(in_h, "float32") / ops.cast(height, "float32")
            scale_w = ops.cast(in_w, "float32") / ops.cast(width, "float32")

            # Create grid with proper alignment
            z_coords = (ops.arange(depth, dtype="float32") + 0.5) * scale_d - 0.5
            y_coords = (ops.arange(height, dtype="float32") + 0.5) * scale_h - 0.5
            x_coords = (ops.arange(width, dtype="float32") + 0.5) * scale_w - 0.5

            # Ensure we don't sample outside the volume
            z_coords = ops.clip(z_coords, 0.0, ops.cast(in_d - 1, "float32"))
            y_coords = ops.clip(y_coords, 0.0, ops.cast(in_h - 1, "float32"))
            x_coords = ops.clip(x_coords, 0.0, ops.cast(in_w - 1, "float32"))

        # Helper function for 1D interpolation
        def interpolate_1d(input_vol, coords, axis):
            # Get floor and ceil indices
            idx0 = ops.cast(ops.floor(coords), "int32")
            idx1 = ops.minimum(idx0 + 1, ops.shape(input_vol)[axis] - 1)

            # Get the values at these indices
            values0 = ops.take(input_vol, idx0, axis=axis)
            values1 = ops.take(input_vol, idx1, axis=axis)

            # Calculate weights
            weight1 = coords - ops.cast(idx0, "float32")
            weight0 = 1.0 - weight1

            # Reshape for broadcasting
            new_shape = [1] * 5  # bs, d, h, w, c
            new_shape[axis] = ops.shape(coords)[0]
            weight0 = ops.reshape(weight0, new_shape)
            weight1 = ops.reshape(weight1, new_shape)

            return weight0 * values0 + weight1 * values1

        # Apply interpolation along each dimension
        interp_d = interpolate_1d(volumes, z_coords, axis=1)
        interp_h = interpolate_1d(interp_d, y_coords, axis=2)
        interp_w = interpolate_1d(interp_h, x_coords, axis=3)

        return ops.cast(interp_w, original_dtype)

    if method == "trilinear":
        return trilinear_resize(volumes, depth, height, width, align_corners)
    else:
        raise ValueError(f"Unsupported resize method: {method}")


def camel_to_snake(name: str) -> str:
    # Step 1: Put underscore between lower-uppercase or digit-uppercase
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    # Step 2: Handle acronym + word boundary (e.g., "CE" + "Loss")
    s2 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s1)
    return s2.lower()


class DescriptionObject:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


class DescribeMixin:
    _skip_keys = {"layers", "input_layers", "output_layers", "weights", "layer_names"}

    @classmethod
    def class_describe(cls, pretty: bool = True):
        """Return class-level description including docstring and __init__ args."""
        base_doc = inspect.cleandoc(cls.__doc__ or "No description available.")
        name = cls.__name__

        if pretty:
            lines = [f"📌 Class: {name}", "\n📝 Description:", base_doc]

            # Optional: show allowed backbones if defined
            if hasattr(cls, "ALLOWED_BACKBONE_FAMILIES"):
                lines.append("\n🧩 Allowed Backbone Families:")
                for fam in cls.ALLOWED_BACKBONE_FAMILIES:
                    lines.append(f"  • {fam}")

            # Constructor arguments
            lines.append("\n⚙️ Constructor Arguments:")
            try:
                sig = inspect.signature(cls.__init__)
                for pname, param in sig.parameters.items():
                    if pname == "self":
                        continue
                    # default value
                    default = (
                        f"= {param.default!r}"
                        if param.default is not inspect.Parameter.empty
                        else ""
                    )
                    # annotation
                    if param.annotation != inspect.Parameter.empty:
                        annot_name = getattr(param.annotation, "__name__", repr(param.annotation))
                        annot = f": {annot_name}"
                    else:
                        annot = ""
                    lines.append(f"  {pname}{annot} {default}".rstrip())

                init_doc = inspect.cleandoc(cls.__init__.__doc__ or "")
                if init_doc:
                    lines.append(f"\n📘 Details Constructor Arguments:\n{init_doc}")
            except Exception:
                lines.append("  <unable to inspect constructor>")

            return DescriptionObject("\n".join(lines))

        # Machine-friendly dict version
        desc = {"name": name, "doc": base_doc}
        if hasattr(cls, "ALLOWED_BACKBONE_FAMILIES"):
            desc["allowed_backbones"] = cls.ALLOWED_BACKBONE_FAMILIES
        try:
            desc["args"] = {
                pname: str(param)
                for pname, param in inspect.signature(cls.__init__).parameters.items()
                if pname != "self"
            }
        except Exception:
            desc["args"] = {}
        return desc

    def _format_param(self, k, v, encoder_desc=None, indent="  "):
        # Special case: encoder key
        if k == "encoder" and encoder_desc is not None:
            lines = [f"{indent}• {encoder_desc['class']}("]
            for nk, nv in encoder_desc["params"].items():
                lines.append(f"{indent}  • {nk}: {nv}")
            lines.append(f"{indent})")
            return "\n".join(lines)
        if isinstance(v, (dict, list, tuple)):
            return f"{indent}• {k}: {v}"
        else:
            return f"{indent}• {k}: {v}"

    def instance_describe(self, pretty: bool = True):
        name = self.__class__.__name__

        # if encoder attribute exists and has instance_describe
        encoder = getattr(self, "encoder", None)
        encoder_desc = None
        if encoder is not None and hasattr(encoder, "instance_describe"):
            encoder_desc = encoder.instance_describe(pretty=False)

        params = {}
        if hasattr(self, "get_config") and callable(self.get_config):
            params = self.get_config()

        # remove unwanted internal keys
        params = {k: v for k, v in params.items() if k not in self._skip_keys}

        if not pretty:
            return {"class": name, "params": params}

        # pretty string
        lines = [f"Instance of {name}"]
        if not params:
            lines.append("  • <no config available>")
        else:
            for k, v in params.items():
                lines.append(self._format_param(k, v, encoder_desc=encoder_desc))
        return DescriptionObject("\n".join(lines))
