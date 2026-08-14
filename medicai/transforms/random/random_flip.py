from typing import Sequence, Union

import keras
import tensorflow as tf

from ..base import RandomTransform, _apply_if_applied, _pop_last_transform_trace
from ..spatial.flip import Flip
from ..tensor_bundle import TensorBundle


class RandomFlip(RandomTransform):
    """Randomly flip selected tensors along specified spatial axes.

    ``RandomFlip`` wraps the deterministic :class:`~medicai.transforms.Flip`
    kernel with probabilistic application. When the sampled Bernoulli draw is
    ``True``, each selected channel-last tensor is reversed along the
    configured axes.

    Depending on ``input_layout``, this transform supports:

    - sample 2D tensors shaped ``(H, W, C)``
    - sample 3D tensors shaped ``(D, H, W, C)``
    - batch 2D tensors shaped ``(B, H, W, C)``
    - batch 3D tensors shaped ``(B, D, H, W, C)``

    Args:
        keys: Keys of the tensors to flip.
        prob: Probability of applying the flip.
        spatial_axis: Spatial axis or axes to reverse. For 2D tensors, ``0``
            refers to the vertical-height axis and ``1`` refers to the
            horizontal-width axis. For 3D tensors using sample-space axis
            numbering ``(D, H, W)``, axis ``0`` is the depth direction, axis
            ``1`` is the height direction, and axis ``2`` is the width
            direction; these correspond to the sagittal, coronal, and axial
            viewing orientations depending on which axis is being mirrored.
        input_layout: Channel-last tensor layout. Supported values are
            ``"HWC"``, ``"DHWC"``, ``"BHWC"``, and ``"BDHWC"``.
        seed: Optional random seed. Supports ``None``, an integer seed, or a
            ``keras.random.SeedGenerator``. The seed controls the Bernoulli
            apply/skip draw for each call. A fresh transform instance created
            with the same integer seed replays the same random sequence, while
            repeated calls on one instance advance its internal seed stream.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Randomly flip a 2D image using a raw Python dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomFlip

            transform = RandomFlip(
                keys=["image"],
                prob=0.5,
                spatial_axis=0,
                input_layout="HWC",
            )
            image = tf.random.normal((64, 64, 1))
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        Randomly flip a 3D image stored in a ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomFlip, TensorBundle

            transform = RandomFlip(
                keys=["image"],
                prob=0.5,
                spatial_axis=0,
                input_layout="DHWC",
            )
            image = tf.random.normal((32, 64, 64, 1))
            bundle = TensorBundle({"image": image})
            result = transform(bundle)
            output = result["image"]
            print(output.shape)
    """

    def __init__(
        self,
        keys: Sequence[str],
        prob: float = 0.1,
        spatial_axis: Union[int, Sequence[int], None] = None,
        *,
        input_layout: str,
        seed: int | keras.random.SeedGenerator | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob, seed=seed)
        self.flip = Flip(
            keys=keys,
            spatial_axis=spatial_axis,
            input_layout=input_layout,
            allow_missing_keys=allow_missing_keys,
        )

    @property
    def invertible(self) -> bool:
        return self.flip.spatial_axis is not None

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        params = self.get_random_params(bundle)
        if not params["enabled"]:
            self.record_skipped(bundle, params)
            return bundle

        return self.apply_with_params(bundle, params)

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        if self.flip.spatial_axis is None:
            return bundle

        trace = self._get_last_random_flip_trace(bundle)
        if trace is None:
            return bundle

        applied = trace.get("applied", False)

        def apply_inverse_flip(tensor: tf.Tensor, _: str) -> tf.Tensor:
            return _apply_if_applied(
                applied,
                lambda tensor=tensor: self.flip.flip_tensor(tensor),
                lambda tensor=tensor: tensor,
            )

        self.flip.apply_to_present_keys(
            bundle, apply_inverse_flip, keys=trace["params"].get("keys", [])
        )
        return bundle

    def get_random_params(self, bundle: TensorBundle) -> dict[str, object]:
        """Sample one Bernoulli decision shared across all selected keys."""
        del bundle
        return {
            "enabled": self.flip.spatial_axis is not None,
            "should_apply": self.sample_should_apply(),
            "spatial_axis": self.flip.spatial_axis,
            "input_layout": self.flip.input_layout,
        }

    def apply_with_params(
        self,
        bundle: TensorBundle,
        params: dict[str, object],
    ) -> TensorBundle:
        """Apply the shared flip kernel conditionally using sampled params."""
        present_keys = self.flip.apply_to_present_keys(
            bundle,
            lambda tensor, key: self.transform_tensor(tensor, key, params),
        )
        self.record_random_transform(
            bundle,
            params=self.build_trace_params(params, present_keys),
            applied=params["should_apply"],
            kernel="Flip",
        )
        return bundle

    def record_skipped(
        self,
        bundle: TensorBundle,
        params: dict[str, object],
    ) -> TensorBundle:
        """Record a skipped random flip when no axes are configured."""
        self.record_random_transform(
            bundle,
            params=self.build_trace_params(params, []),
            applied=False,
            kernel="Flip",
        )
        return bundle

    def transform_tensor(
        self,
        tensor: tf.Tensor,
        key: str,
        params: dict[str, object],
    ) -> tf.Tensor:
        """Apply the sampled flip decision to one tensor."""
        del key
        return _apply_if_applied(
            params["should_apply"],
            lambda tensor=tensor: self.flip.flip_tensor(
                tensor,
                spatial_axis=params["spatial_axis"],
            ),
            lambda tensor=tensor: tensor,
        )

    def build_trace_params(
        self,
        params: dict[str, object],
        present_keys: Sequence[str],
    ) -> dict[str, object]:
        """Build random trace metadata for the current flip."""
        return {
            "keys": list(present_keys),
            "spatial_axis": params["spatial_axis"],
            "input_layout": params["input_layout"],
        }

    def _get_last_random_flip_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)
