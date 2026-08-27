"""Placeholder for Torch end-to-end training coverage.

Backend-specific tests will be added after the migrated transform set and
training scenarios are finalized.
"""

import pytest


@pytest.mark.integration
@pytest.mark.skip(reason="Torch training scenarios are not defined yet.")
def test_torch_training_placeholder():
    """Reserve the test entry point for Torch training coverage."""

'''
Plans:
1. train with torch.data API:
    - dummy 2D dataset (clas, Seg), dummy 3d dataset (clas, Seg)

2. train with tf.data API:
    - dummy 2D dataset (clas, Seg), dummy 3d dataset (clas, Seg)

3. train with pygrain API:
    - dummy 2D dataset (clas, Seg), dummy 3d dataset (clas, Seg)
'''
