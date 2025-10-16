from keras import ops


def resize_volumes(volumes, depth, height, width, method="trilinear", align_corners=False):
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
