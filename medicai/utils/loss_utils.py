from keras import ops


def soft_erode(inputs):
    ndim = len(inputs.shape)

    if ndim == 4:
        p1 = -ops.max_pool(-inputs, pool_size=(3, 1), strides=1, padding="same")
        p2 = -ops.max_pool(-inputs, pool_size=(1, 3), strides=1, padding="same")
        return ops.minimum(p1, p2)

    elif ndim == 5:
        p1 = -ops.max_pool(-inputs, pool_size=(3, 1, 1), strides=1, padding="same")
        p2 = -ops.max_pool(-inputs, pool_size=(1, 3, 1), strides=1, padding="same")
        p3 = -ops.max_pool(-inputs, pool_size=(1, 1, 3), strides=1, padding="same")
        return ops.minimum(ops.minimum(p1, p2), p3)

    else:
        raise ValueError(f"soft_erode only supports 2D or 3D inputs, got shape {inputs.shape}")


def soft_dilate(inputs):
    return ops.max_pool(inputs, pool_size=3, strides=1, padding="same")


def soft_open(inputs):
    inputs = soft_erode(inputs)
    inputs = soft_dilate(inputs)
    return inputs


def soft_skeletonize(inputs, iters):
    inputs_open = soft_open(inputs)
    skel = ops.relu(inputs - inputs_open)
    for _ in range(iters):
        inputs = soft_erode(inputs)
        inputs_open = soft_open(inputs)
        delta = ops.relu(inputs - inputs_open)
        intersect = ops.multiply(skel, delta)
        skel = skel + ops.relu(delta - intersect)
    return skel
