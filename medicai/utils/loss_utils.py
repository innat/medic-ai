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
    """
    Computes a differentiable approximation of the morphological skeleton of a
    binary or probabilistic segmentation mask.

    Soft skeletonization iteratively removes object boundaries using soft
    morphological erosion while preserving the central structure (skeleton)
    of the object. Unlike traditional skeletonization algorithms, this
    implementation is fully differentiable and can be used inside deep learning
    training pipelines.

    The algorithm follows an iterative thinning process:

    1. Apply soft opening (erosion followed by dilation).
    2. Extract boundary-free center regions.
    3. Iteratively erode the mask.
    4. Accumulate newly discovered skeleton components.
    5. Merge all skeleton fragments into the final skeleton representation.

    This operation is commonly used in:

    - Topology-aware segmentation losses
    - Connectivity-preserving objectives
    - Vessel segmentation
    - Road extraction
    - Tubular structure analysis
    - ClDice and skeleton-based segmentation metrics

    Args:
        inputs (Tensor):
            Input segmentation mask or probability map.

            Supported shapes:

            - 2D: ``(batch_size, height, width, channels)``
            - 3D: ``(batch_size, depth, height, width, channels)``

            Values are typically expected to be in the range ``[0, 1]``.

        iters (int):
            Number of iterative erosion steps used to construct the skeleton. Larger values produce more complete skeletons for larger objects 
            but increase computational cost.

    Returns:
        Tensor:
            Soft skeleton representation with the same shape and dtype as
            the input tensor.

            Output shape:
            - 2D: ``(batch_size, height, width, channels)``
            - 3D: ``(batch_size, depth, height, width, channels)``

    Raises:
        ValueError:
            If the input tensor is not a supported 2D or 3D segmentation tensor.

    Example:
        2D skeletonization::

            import numpy as np
            from keras import ops
            from medicai.utils import soft_skeletonize

            mask = np.random.randint(0, 2, size=(1, 256, 256, 1))
            mask = ops.convert_to_tensor(mask)
            skeleton = soft_skeletonize(mask, iters=10)
            print(skeleton.shape) # (1, 256, 256, 1)

        3D skeletonization::

            import numpy as np
            from keras import ops
            from medicai.utils import soft_skeletonize

            mask = np.random.randint(0, 2, size=(1, 64, 128, 128, 1))
            mask = ops.convert_to_tensor(mask)
            skeleton = soft_skeletonize(mask, iters=10)
            print(skeleton.shape) # (1, 64, 128, 128, 1)

    References:
        - clDice - A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation.
          `arXiv:2003.07311 <https://arxiv.org/abs/2003.07311>`_
    """
    inputs_open = soft_open(inputs)
    skel = ops.relu(inputs - inputs_open)
    for _ in range(iters):
        inputs = soft_erode(inputs)
        inputs_open = soft_open(inputs)
        delta = ops.relu(inputs - inputs_open)
        intersect = ops.multiply(skel, delta)
        skel = skel + ops.relu(delta - intersect)
    return skel
