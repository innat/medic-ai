"""Placeholder for Torch end-to-end training coverage.

Torch single-device training will be added first. Multi-device distribution
is intentionally deferred until Keras exposes a stable Torch distribution
API for this project.
"""

import pytest


@pytest.mark.integration
@pytest.mark.skip(reason="Torch training scenarios are not defined yet.")
def test_torch_training_placeholder():
    """Reserve the entry point for Torch single-device training coverage."""

'''
Plans:
1. train with torch.data API:
    - dummy 2D dataset (clas, Seg), dummy 3d dataset (clas, Seg)

2. train with tf.data API:
    - dummy 2D dataset (clas, Seg), dummy 3d dataset (clas, Seg)

3. train with pygrain API:
    - dummy 2D dataset (clas, Seg), dummy 3d dataset (clas, Seg)
'''
