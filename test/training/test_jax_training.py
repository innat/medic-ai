"""Placeholder for JAX end-to-end training coverage.

JAX training should use Keras distribution APIs rather than backend-private
distribution code once the shared scenarios are finalized.
"""

import pytest


@pytest.mark.integration
@pytest.mark.skip(reason="JAX training scenarios are not defined yet.")
def test_jax_training_placeholder():
    """Reserve the entry point for Keras-distributed JAX training coverage."""

'''
Plans:
1. train with torch.data API:
    - dummy 2D dataset (clas, Seg), dummy 3d dataset (clas, Seg)

2. train with tf.data API:
    - dummy 2D dataset (clas, Seg), dummy 3d dataset (clas, Seg)

3. train with pygrain API:
    - dummy 2D dataset (clas, Seg), dummy 3d dataset (clas, Seg)
'''
