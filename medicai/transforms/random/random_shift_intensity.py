from numbers import Number
from typing import Any, Sequence, Tuple, Union

import keras
from keras import ops

from ..base import (
    RandomTransform,
    _apply_if_applied,
    _normalize_keys,
    _pop_last_transform_trace,
)
from ..intensity.shift_intensity import ShiftIntensity
from ..tensor_bundle import TensorBundle
from ..utils import get_tensor_rank, resolve_input_layout


class RandomShiftIntensity(RandomTransform):
    """Randomly shift intensity values of selected tensors.

    ``RandomShiftIntensity`` samples additive offsets from a configured range
    and applies them with a given probability using the deterministic
    :class:`~medicai.transforms.ShiftIntensity` kernel.

    The transform expects channel-last tensors such as ``(H, W, C)`` or
    ``(D, H, W, C)``. Depending on ``input_layout``, it supports:

    - sample 2D tensors shaped ``(H, W, C)``
    - sample 3D tensors shaped ``(D, H, W, C)``
    - batch 2D tensors shaped ``(B, H, W, C)``
    - batch 3D tensors shaped ``(B, D, H, W, C)``

    Offsets may be sampled once per tensor or separately per channel depending
    on ``channel_wise``. During inversion, the transform uses the sampled
    offsets stored in the transform trace and applies the inverse only to the
    traced keys.

    In the current migrated dual-mode path, if this transform is used on a
    batched tensor through a batch-capable wrapper setup, the sampled offset
    values are shared across the full batch rather than drawn independently
    per batch item.

    Args:
        keys: Keys of the tensors to shift.
        offset: Symmetric scalar magnitude or explicit ``(min, max)`` offset
            range to sample from.
        prob: Probability of applying the shift.
        channel_wise: If ``True``, sample independent per-channel offsets.
        input_layout: Channel-last tensor layout. Supported values are
            ``"HWC"``, ``"DHWC"``, ``"BHWC"``, and ``"BDHWC"``.
        seed: Optional random seed. Supports ``None``, an integer seed, or a
            ``keras.random.SeedGenerator``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:

        TensorFlow backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import RandomShiftIntensity

            transform = RandomShiftIntensity(
                keys=["image"],
                offset=0.1,
                prob=0.5,
                input_layout="HWC",
            )
            image = tf.random.normal((64, 64, 1))
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        JAX backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "jax"

            import jax
            from medicai.transforms import RandomShiftIntensity

            transform = RandomShiftIntensity(
                keys=["image"],
                offset=0.1,
                prob=0.5,
                input_layout="DHWC",
            )
            image = jax.random.normal(
                jax.random.PRNGKey(7), shape=(32, 64, 64, 1)
            )
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        Torch backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "torch"

            import torch
            from medicai.transforms import RandomShiftIntensity

            transform = RandomShiftIntensity(
                keys=["image"], offset=0.1, prob=0.5, input_layout="BHWC"
            )
            torch.manual_seed(7)
            batch = torch.randn((2, 64, 64, 1))
            result = transform({"image": batch})
            output = result["image"]
            print(output.shape)
    """

    def __init__(
        self,
        keys: Sequence[str],
        offset: Union[float, Tuple[float, float]],
        prob: float = 0.1,
        channel_wise: bool = False,
        *,
        input_layout: str,
        seed: int | keras.random.SeedGenerator | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob, seed=seed)
        self.keys = _normalize_keys(keys)
        if isinstance(offset, (int, float)):
            self.offset = (-abs(offset), abs(offset))
        else:
            self.offset = (min(offset), max(offset))

        self.channel_wise = channel_wise
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            transform_name=type(self).__name__,
        )
        self.allow_missing_keys = allow_missing_keys
        self.shift = ShiftIntensity(
            keys=self.keys,
            offset=0.0,
            input_layout=self.input_layout,
            allow_missing_keys=self.allow_missing_keys,
        )

    @property
    def invertible(self) -> bool:
        return True

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        params = self.get_random_params(bundle)
        return self.apply_with_params(bundle, params)

    def get_random_params(self, bundle: TensorBundle) -> dict[str, object]:
        """Sample the Bernoulli decision shared across selected keys."""
        del bundle
        return {
            "should_apply": self.sample_should_apply(),
            "channel_wise": self.channel_wise,
            "offset": self.offset,
            "input_layout": self.input_layout,
        }

    def apply_with_params(
        self,
        bundle: TensorBundle,
        params: dict[str, object],
    ) -> TensorBundle:
        """Apply the sampled shift configuration to all selected keys."""
        sampled_offsets = {}
        present_keys = self.shift.iter_present_keys(bundle)

        def apply_shift(tensor, key: str):
            if params["channel_wise"]:
                offset_shape = [1] * (get_tensor_rank(tensor) - 1) + [tensor.shape[-1]]
                offsets = self.random_uniform(
                    shape=offset_shape,
                    minval=params["offset"][0],
                    maxval=params["offset"][1],
                    dtype=tensor.dtype,
                )
            else:
                offsets = self.random_uniform(
                    shape=(),
                    minval=params["offset"][0],
                    maxval=params["offset"][1],
                    dtype=tensor.dtype,
                )
            sampled_offsets[key] = offsets
            return _apply_if_applied(
                params["should_apply"],
                lambda tensor=tensor, offsets=offsets: self.shift.shift_tensor(
                    tensor, offset=offsets
                ),
                lambda tensor=tensor: tensor,
            )

        self.shift.apply_to_present_keys(bundle, apply_shift, keys=present_keys)
        self.record_random_transform(
            bundle,
            params=self.build_trace_params(params, sampled_offsets),
            applied=params["should_apply"],
            kernel="ShiftIntensity",
        )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = self._get_last_random_shift_trace(bundle)
        if trace is None:
            return bundle

        applied = trace.get("applied", False)
        sampled_offsets = trace["params"].get("sampled_offsets", {})

        def apply_inverse_shift(tensor, key: str):
            offset = sampled_offsets.get(key)
            if offset is None:
                return tensor

            inverse_offset = (
                -offset
                if isinstance(offset, Number)
                else -ops.cast(offset, tensor.dtype)
            )
            return _apply_if_applied(
                applied,
                lambda tensor=tensor, offset=inverse_offset: self.shift.shift_tensor(
                    tensor, offset=offset
                ),
                lambda tensor=tensor: tensor,
            )

        self.shift.apply_to_present_keys(
            bundle,
            apply_inverse_shift,
            keys=trace["params"].get("keys", []),
        )
        return bundle

    def build_trace_params(
        self,
        params: dict[str, object],
        sampled_offsets: dict[str, Any],
    ) -> dict[str, object]:
        """Build random trace metadata for the current intensity shift."""
        return {
            "keys": list(sampled_offsets.keys()),
            "channel_wise": params["channel_wise"],
            "offset": params["offset"],
            "input_layout": params["input_layout"],
            "sampled_offsets": sampled_offsets,
        }

    def _get_last_random_shift_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)
