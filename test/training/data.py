"""Small, deterministic fixtures shared by end-to-end training tests."""

import numpy as np


def make_classification_2d_samples(num_samples: int = 4):
    """Return 2D channel-last images and binary classification targets."""
    images = np.linspace(0.0, 1.0, num=num_samples * 12 * 12, dtype=np.float32).reshape(
        num_samples, 12, 12, 1
    )
    labels = np.asarray([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
    return images, labels[:num_samples]


def make_classification_3d_samples(num_samples: int = 4):
    """Return 3D channel-last volumes and binary classification targets."""
    images = np.linspace(0.0, 1.0, num=num_samples * 6 * 6 * 6, dtype=np.float32).reshape(
        num_samples, 6, 6, 6, 1
    )
    labels = np.asarray([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
    return images, labels[:num_samples]


def make_segmentation_2d_samples(num_samples: int = 4):
    """Return 2D channel-last images and aligned binary masks."""
    images = np.linspace(0.0, 1.0, num=num_samples * 12 * 12, dtype=np.float32).reshape(
        num_samples, 12, 12, 1
    )
    labels = (images > 0.5).astype(np.float32)
    return images, labels


def make_segmentation_3d_samples(num_samples: int = 4):
    """Return 3D channel-last volumes and aligned binary masks."""
    images = np.linspace(0.0, 1.0, num=num_samples * 6 * 6 * 6, dtype=np.float32).reshape(
        num_samples, 6, 6, 6, 1
    )
    labels = (images > 0.5).astype(np.float32)
    return images, labels
