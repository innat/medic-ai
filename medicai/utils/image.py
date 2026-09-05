from keras import ops


def _cubic_bspline_weights(fraction):
    one_minus = 1.0 - fraction
    return (
        one_minus**3 / 6.0,
        (3.0 * fraction**3 - 6.0 * fraction**2 + 4.0) / 6.0,
        (-3.0 * fraction**3 + 3.0 * fraction**2 + 3.0 * fraction + 1.0) / 6.0,
        fraction**3 / 6.0,
    )


def _map_bspline_indices(indices, size, boundary):
    if boundary == "nearest":
        return ops.clip(indices, 0, size - 1), ops.ones_like(indices, dtype="bool")
    if boundary == "wrap":
        return ops.mod(indices, size), ops.ones_like(indices, dtype="bool")
    if boundary == "reflect":
        if size == 1:
            return ops.zeros_like(indices), ops.ones_like(indices, dtype="bool")
        period = 2 * (size - 1)
        reflected = ops.mod(ops.abs(indices), period)
        reflected = ops.where(reflected <= size - 1, reflected, period - reflected)
        return reflected, ops.ones_like(indices, dtype="bool")
    if boundary == "constant":
        valid = ops.logical_and(indices >= 0, indices < size)
        return ops.clip(indices, 0, size - 1), valid
    raise ValueError("boundary must be 'nearest', 'reflect', 'wrap', or 'constant'.")


def _resample_bspline_axis(field, axis, target_size, boundary, fill_value):
    input_size = int(field.shape[axis + 1])
    denominator = max(target_size - 1, 1)
    numerator = max(input_size - 1, 0)
    coordinates = ops.arange(target_size, dtype="float32") * (numerator / denominator)
    base = ops.floor(coordinates)
    fraction = coordinates - base
    values = []
    for offset, weight in zip((-1, 0, 1, 2), _cubic_bspline_weights(fraction)):
        indices, valid = _map_bspline_indices(
            ops.cast(base + offset, "int32"), input_size, boundary
        )
        sample = ops.take(field, indices, axis=axis + 1)
        reshape = [1] * len(field.shape)
        reshape[axis + 1] = target_size
        weight = ops.reshape(weight, reshape)
        valid = ops.reshape(valid, reshape)
        if boundary == "constant":
            sample = ops.where(valid, sample, ops.cast(fill_value, field.dtype))
        values.append(sample * weight)
    return sum(values[1:], values[0])


def _resample_bspline_field(field, target_shape, boundary, fill_value):
    result = field
    for axis, target_size in enumerate(target_shape):
        result = _resample_bspline_axis(
            result, axis, target_size, boundary, fill_value
        )
    return result


def _resample_linear_axis(field, axis, target_size, align_corners):
    """Apply a two-tap linear interpolation pass along one spatial axis."""
    input_size = int(field.shape[axis + 1])
    if align_corners:
        coordinates = ops.linspace(
            0.0,
            float(max(input_size - 1, 0)),
            target_size,
        )
    else:
        scale = float(input_size) / float(target_size)
        coordinates = (ops.arange(target_size, dtype="float32") + 0.5) * scale - 0.5
        coordinates = ops.clip(coordinates, 0.0, float(max(input_size - 1, 0)))
    index0 = ops.cast(ops.floor(coordinates), "int32")
    index1 = ops.minimum(index0 + 1, input_size - 1)
    values0 = ops.take(field, index0, axis=axis + 1)
    values1 = ops.take(field, index1, axis=axis + 1)
    reshape = [1] * len(field.shape)
    reshape[axis + 1] = target_size
    weight = ops.reshape(coordinates - ops.cast(index0, "float32"), reshape)
    return values0 * (1.0 - weight) + values1 * weight


def _resample_linear_field(field, target_shape, align_corners):
    result = field
    for axis, target_size in enumerate(target_shape):
        result = _resample_linear_axis(result, axis, target_size, align_corners)
    return result


def resample_displacement_field(
    field,
    target_shape,
    method="trilinear",
    *,
    boundary="nearest",
    fill_value=0.0,
    align_corners=False,
):
    """Resample a 2D or 3D channel-last displacement field.

    The ``"bspline"`` method evaluates a tensor-product cubic B-spline field
    from regularly spaced control-point coefficients. For a cubic spline, each
    output coordinate uses at most four neighboring control points along each
    spatial axis; the implementation applies those four-tap evaluations
    separably across the spatial dimensions. This follows the standard
    B-spline basis convention described by SciPy's
    :class:`scipy.interpolate.BSpline`, but uses ``keras.ops`` so it remains
    backend agnostic at runtime.

    B-spline field interpolation is distinct from image interpolation. The
    resulting displacement field is later used to sample images or labels;
    ``boundary`` controls control-point field resampling, while a transform's
    image ``fill_mode`` controls sampling outside the image domain. The
    ``"reflect"`` boundary uses the edge-non-repeating mirror convention.

    The B-spline path treats the input values as control-point coefficients.
    It does not prefilter arbitrary sampled values to obtain interpolation
    coefficients, and it does not imply cubic interpolation of the image or
    label tensor itself.

    Args:
        field: Tensor shaped ``(B, H, W, 2)`` or ``(B, D, H, W, 3)``.
        target_shape: Target spatial shape, ``(H, W)`` or ``(D, H, W)``.
        method: ``"bilinear"`` for 2D, ``"trilinear"`` for 3D, or
            ``"bspline"`` for cubic B-spline field interpolation. The linear
            methods use the existing backend-neutral volume/axis resizing
            semantics; ``"bspline"`` uses control-grid-aligned coordinates.
        boundary: Boundary mode for B-spline interpolation. Supported values
            are ``"nearest"``, ``"reflect"``, ``"wrap"``, and ``"constant"``.
            This is separate from an image transform's ``fill_mode``.
        fill_value: Constant boundary value when ``boundary="constant"``.
        align_corners: Coordinate convention for linear interpolation. The
            B-spline path uses control-grid-aligned coordinates.

    Returns:
        Tensor: The resampled displacement field with the requested shape.

    Raises:
        ValueError: If the field rank, target rank, or method is invalid.

    Example:
        Resample a coarse 3D displacement field before using it to warp a
        volume::

            from medicai.utils import resample_displacement_field
            from keras import ops

            coarse_field = ops.zeros((1, 20, 32, 32, 3), dtype="float32")

            dense_field = resample_displacement_field(
                coarse_field,
                target_shape=(160, 256, 256),
                method="bspline",
                boundary="nearest",
            )
            print(dense_field.shape)  # (1, 160, 256, 256, 3)

            coarse_field_2d = ops.zeros((1, 32, 32, 2), dtype="float32")
            dense_field_2d = resample_displacement_field(
                coarse_field_2d,
                target_shape=(224, 224),
                method="bspline",
                boundary="nearest",
            )
            print(dense_field_2d.shape)  # (1, 224, 224, 2)
    """
    rank = len(target_shape)
    expected_rank = rank + 2
    if rank not in (2, 3) or len(field.shape) != expected_rank:
        raise ValueError(
            "`field` and `target_shape` must describe a 2D or 3D channel-last "
            "field."
        )
    if method == "bspline":
        if boundary not in {"nearest", "reflect", "wrap", "constant"}:
            raise ValueError(
                "B-spline `boundary` must be 'nearest', 'reflect', 'wrap', or "
                "'constant'."
            )
        return _resample_bspline_field(field, target_shape, boundary, fill_value)
    if method == "bilinear" and rank == 2:
        return _resample_linear_field(field, target_shape, align_corners)
    if method == "trilinear" and rank == 3:
        return resize_volumes(
            field,
            depth=target_shape[0],
            height=target_shape[1],
            width=target_shape[2],
            method="trilinear",
            align_corners=align_corners,
        )
    raise ValueError(
        f"Unsupported displacement-field method {method!r} for {rank}D input. "
        "Use 'bilinear' for 2D, 'trilinear' for 3D, or 'bspline'."
    )


def resize_volumes(volumes, depth, height, width, method="trilinear", align_corners=False):
    """
    Resizes 5D volumetric tensors using either trilinear interpolation or nearest-neighbor sampling. This function
    provides a backend-agnostic implementation of ``3D`` resizing for tensors shaped as
    ``(batch, depth, height, width, channels)``. It supports two interpolation strategies:

    1. **Trilinear interpolation**:

       - Smooth, differentiable resizing method
       - Performs sequential linear interpolation along depth, height, and width axes
       - PyTorch-compatible coordinate mapping (optional ``align_corners`` behavior)

    2. **Nearest-neighbor interpolation**:

       - Fast, non-differentiable approximation
       - Selects closest voxel indices without interpolation

    The implementation is designed for deep learning frameworks using Keras 3
    backend abstraction (`keras.ops`), making it compatible with TensorFlow, JAX,
    and PyTorch backends.

    Args:
        volumes (Tensor): Input ``5D`` tensor of shape: ``(batch_size, depth, height, width, channels)``.
        depth (int): Target depth dimension after resizing.
        height (int): Target height dimension after resizing.
        width (int): Target width dimension after resizing.
        method (str, optional): Interpolation method to use for resizing. Supported values:

            - ``"trilinear"``: Continuous interpolation (default)
            - ``"nearest"``: Discrete nearest-neighbor sampling

        align_corners (bool, optional): Only used when ``method="trilinear"``.

            If ``True``:
                - Corners of input and output tensors are exactly aligned.
                - Coordinates are mapped from edge-to-edge.

            If ``False``:
                - Uses PyTorch-style scaling with half-pixel offset alignment.
                - Generally produces smoother and more stable resizing behavior.

    Examples:
        .. code-block:: python

            import torch
            from medicai.utils import resize_volumes

            x = torch.rand(1, 96, 96, 96, 3)
            output = resize_volumes(
                x,
                depth=128,
                height=128,
                width=128,
                method='trilinear',
                align_corners=False
            )
            print(output.shape) # (1, 128, 128, 128, 3)
    """

    def trilinear_resize(volumes, depth, height, width, align_corners):
        original_dtype = volumes.dtype
        volumes = ops.cast(volumes, "float32")
        in_d = ops.shape(volumes)[1]
        in_h = ops.shape(volumes)[2]
        in_w = ops.shape(volumes)[3]

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

    def nearest(volumes, depth, height, width):
        shape = ops.shape(volumes)
        bs, d, h, w, c = shape[0], shape[1], shape[2], shape[3], shape[4]

        z = ops.linspace(0.0, ops.cast(d - 1, "float32"), depth)
        z = ops.cast(ops.round(z), "int32")
        z = ops.clip(z, 0, d - 1)

        y = ops.linspace(0.0, ops.cast(h - 1, "float32"), height)
        y = ops.cast(ops.round(y), "int32")
        y = ops.clip(y, 0, h - 1)

        x = ops.linspace(0.0, ops.cast(w - 1, "float32"), width)
        x = ops.cast(ops.round(x), "int32")
        x = ops.clip(x, 0, w - 1)

        # Create 3D grid
        Z, Y, X = ops.meshgrid(z, y, x, indexing="ij")

        # indices
        Z = ops.reshape(Z, (-1,))
        Y = ops.reshape(Y, (-1,))
        X = ops.reshape(X, (-1,))

        # Batch replication
        batch_idx = ops.repeat(ops.arange(bs), ops.shape(Z)[0])
        Z = ops.tile(Z, [bs])
        Y = ops.tile(Y, [bs])
        X = ops.tile(X, [bs])

        # Flatten input
        flat = ops.reshape(volumes, (bs * d * h * w, c))

        # Compute linear indices
        indices = (batch_idx * d * h * w) + (Z * h * w) + (Y * w) + X
        result = ops.take(flat, indices, axis=0)

        # Reshape to final size
        result = ops.reshape(result, (bs, depth, height, width, c))
        return result

    if method == "trilinear":
        return trilinear_resize(volumes, depth, height, width, align_corners)

    elif method == "nearest":
        return nearest(volumes, depth, height, width)

    else:
        raise ValueError(f"Unsupported resize method: {method}")
