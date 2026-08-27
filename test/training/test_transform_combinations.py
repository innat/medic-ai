"""Backend-neutral integration coverage for composed migrated transforms."""

import numpy as np
import pytest
from keras import ops

from medicai.transforms import Compose, Flip, RandomChoice, Rotate90, SpatialCrop


@pytest.mark.integration
def test_geometric_transform_chain_preserves_image_label_alignment():
    """Compose crop, flip, rotation, and random choice without desynchronizing labels."""
    image = np.zeros((12, 12, 1), dtype="float32")
    image[2:10, 3:9, 0] = 1.0
    label = image.copy()

    pipeline = Compose(
        [
            SpatialCrop(
                keys=["image", "label"],
                crop_size=(8, 8),
                crop_start=(2, 2),
                input_layout="HWC",
            ),
            Flip(keys=["image", "label"], spatial_axis=0, input_layout="HWC"),
            Rotate90(
                keys=["image", "label"],
                k=1,
                spatial_axis=(0, 1),
                input_layout="HWC",
            ),
            RandomChoice(
                transforms=[
                    Flip(keys=["image", "label"], spatial_axis=0, input_layout="HWC"),
                    Rotate90(
                        keys=["image", "label"],
                        k=2,
                        spatial_axis=(0, 1),
                        input_layout="HWC",
                    ),
                ],
                num_choices=1,
                prob=1.0,
                seed=7,
            ),
        ]
    )

    result = pipeline({"image": image, "label": label})

    np.testing.assert_allclose(
        ops.convert_to_numpy(result["image"]),
        ops.convert_to_numpy(result["label"]),
    )
    assert tuple(ops.shape(result["image"])) == (8, 8, 1)


@pytest.mark.integration
def test_composed_geometry_can_be_inverted_to_original_canvas():
    """Placement inversion restores the original canvas after a deterministic crop."""
    image = ops.convert_to_tensor(np.arange(12 * 12, dtype="float32").reshape(12, 12, 1))
    pipeline = Compose(
        [
            SpatialCrop(
                keys=["image"],
                crop_size=(8, 8),
                crop_start=(2, 2),
                input_layout="HWC",
            ),
            Flip(keys=["image"], spatial_axis=1, input_layout="HWC"),
        ]
    )

    result = pipeline({"image": image})
    restored = pipeline.inverse(result)

    assert tuple(ops.shape(restored["image"])) == (12, 12, 1)
