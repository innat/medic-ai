import numpy as np
import pytest

from medicai.utils.inference import (
    extract_patches,
    merge_patches,
    predict_patches,
    sliding_window_inference,
    sliding_window_inference_old,
)


class MockModel:
    """Deterministic mock model supporting arbitrary spatial dimensionality."""

    def __init__(self, output_classes=2, num_spatial_dims=3, fail_on_predict=False):
        self.output_classes = output_classes
        self.fail_on_predict = fail_on_predict
        # Build output_shape dynamically to match expected input rank
        # (batch, *spatial, channels)
        self.output_shape = (None,) * (num_spatial_dims + 1) + (self.output_classes,)

    def predict(self, x, verbose=0):
        if self.fail_on_predict:
            raise RuntimeError("Injected model failure")
        output_shape = list(x.shape)
        output_shape[-1] = self.output_classes
        res = np.ones(output_shape, dtype=np.float32)
        res = res * np.mean(x, axis=-1, keepdims=True) + 1.0
        return res


@pytest.mark.parametrize("overlap", [0.25, 0.5])
@pytest.mark.parametrize("mode", ["constant", "gaussian"])
@pytest.mark.parametrize("sigma_scale", [0.125, 0.25])
def test_3d_sliding_window_inference_equivalence(overlap, mode, sigma_scale):
    np.random.seed(42)
    inputs = np.random.rand(1, 32, 32, 32, 1).astype(np.float32)

    model = MockModel(output_classes=2, num_spatial_dims=3)
    roi_size = (16, 16, 16)
    sw_batch_size = 4
    num_classes = 2

    # Case 0: original monolithic function
    output_case_0 = sliding_window_inference_old(
        inputs=inputs,
        model=model,
        num_classes=num_classes,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode=mode,
        sigma_scale=sigma_scale,
    )

    # Case 1: new wrapper
    output_case_1 = sliding_window_inference(
        inputs=inputs,
        model=model,
        num_classes=num_classes,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode=mode,
        sigma_scale=sigma_scale,
    )

    # Case 2: step-by-step with generator API
    padded_inputs, info = extract_patches(
        inputs=inputs,
        roi_size=roi_size,
        overlap=overlap,
        mode=mode,
        sigma_scale=sigma_scale,
    )
    pred_gen = predict_patches(
        padded_inputs=padded_inputs,
        info=info,
        model=model,
        sw_batch_size=sw_batch_size,
    )
    output_case_2 = merge_patches(
        patch_generator=pred_gen,
        info=info,
        num_classes=num_classes,
    )

    np.testing.assert_allclose(
        output_case_0,
        output_case_1,
        rtol=1e-5,
        atol=1e-5,
        err_msg="3D: Wrapper output does not match old monolithic!",
    )
    np.testing.assert_allclose(
        output_case_0,
        output_case_2,
        rtol=1e-5,
        atol=1e-5,
        err_msg="3D: Component pipeline output does not match old monolithic!",
    )


@pytest.mark.parametrize("overlap", [0.25, 0.5])
@pytest.mark.parametrize("mode", ["constant", "gaussian"])
@pytest.mark.parametrize("sigma_scale", [0.125, 0.25])
def test_2d_sliding_window_inference_equivalence(overlap, mode, sigma_scale):
    np.random.seed(42)
    inputs = np.random.rand(1, 64, 64, 1).astype(np.float32)

    model = MockModel(output_classes=2, num_spatial_dims=2)
    roi_size = (32, 32)
    sw_batch_size = 4
    num_classes = 2

    # Case 1: new wrapper
    output_case_1 = sliding_window_inference(
        inputs=inputs,
        model=model,
        num_classes=num_classes,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode=mode,
        sigma_scale=sigma_scale,
    )

    # Case 2: step-by-step with generator API
    padded_inputs, info = extract_patches(
        inputs=inputs,
        roi_size=roi_size,
        overlap=overlap,
        mode=mode,
        sigma_scale=sigma_scale,
    )
    pred_gen = predict_patches(
        padded_inputs=padded_inputs,
        info=info,
        model=model,
        sw_batch_size=sw_batch_size,
    )
    output_case_2 = merge_patches(
        patch_generator=pred_gen,
        info=info,
        num_classes=num_classes,
    )

    assert output_case_1.shape == (
        1,
        64,
        64,
        2,
    ), f"Unexpected 2D output shape: {output_case_1.shape}"
    np.testing.assert_allclose(
        output_case_1,
        output_case_2,
        rtol=1e-5,
        atol=1e-5,
        err_msg="2D: Wrapper output does not match step-by-step pipeline!",
    )


def test_num_classes_none_fallback_3d():
    """Verify num_classes=None fallback uses model.output_shape[-1]."""
    np.random.seed(42)
    inputs = np.random.rand(1, 32, 32, 32, 1).astype(np.float32)

    model = MockModel(output_classes=3, num_spatial_dims=3)
    roi_size = (16, 16, 16)
    sw_batch_size = 4

    output = sliding_window_inference(
        inputs=inputs,
        model=model,
        num_classes=None,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
    )
    assert output.shape == (1, 32, 32, 32, 3)


def test_num_classes_none_fallback_2d():
    """Verify num_classes=None fallback uses model.output_shape[-1]."""
    np.random.seed(42)
    inputs = np.random.rand(1, 64, 64, 1).astype(np.float32)

    model = MockModel(output_classes=4, num_spatial_dims=2)
    roi_size = (32, 32)
    sw_batch_size = 4

    output = sliding_window_inference(
        inputs=inputs,
        model=model,
        num_classes=None,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
    )
    assert output.shape == (1, 64, 64, 4)


# ------------------------------------------------------------------ #
#  Component and Error Handling Tests                                #
# ------------------------------------------------------------------ #


def test_extract_patches_metadata():
    """Verify that extract_patches computes correct padded shapes and metadata."""
    inputs = np.zeros((1, 10, 10, 1))
    roi_size = (16, 16)
    padded_inputs, info = extract_patches(inputs, roi_size)

    # Padding should make it (1, 16, 16, 1) since inputs < roi_size
    assert padded_inputs.shape == (1, 16, 16, 1)
    assert info["original_image_size"] == (10, 10)
    assert info["padded_image_size"] == (16, 16)
    assert info["roi_size"] == (16, 16)
    assert info["num_spatial_dims"] == 2
    assert info["batch_size"] == 1


def test_empty_slices_error():
    """Verify that merge_patches raises ValueError if generator is empty."""
    # This shouldn't happen with normal inputs, but we test the guard
    info = {
        "batch_size": 1,
        "padded_image_size": (16, 16),
        "original_image_size": (16, 16),
        "pad_size": [[0, 0], [0, 0], [0, 0]],
    }

    def empty_gen():
        if False:
            yield None

    with pytest.raises(ValueError, match="patch_generator yielded no batches"):
        merge_patches(empty_gen(), info)


def test_spatial_shape_mismatch():
    """Verify error when model output spatial dims != roi_size."""

    class BrokenShapeModel:
        def predict(self, x, verbose=0):
            # Return half spatial size
            bs = x.shape[0]
            return np.zeros((bs, 8, 8, 2))

    inputs = np.zeros((1, 16, 16, 1))
    roi_size = (16, 16)
    with pytest.raises(RuntimeError, match="requires the model to preserve spatial dimensions"):
        sliding_window_inference(inputs, BrokenShapeModel(), 2, roi_size, 1)


def test_model_failure_context():
    """Verify that failures in predict_patches are wrapped with context."""
    inputs = np.zeros((1, 16, 16, 1))
    model = MockModel(fail_on_predict=True, num_spatial_dims=2)

    with pytest.raises(RuntimeError, match="failed during the prediction/merging phase"):
        sliding_window_inference(inputs, model, 2, (16, 16), 1)


def test_input_rank_validation():
    """Verify rank check."""
    with pytest.raises(ValueError, match="got rank 2"):
        sliding_window_inference(np.zeros((10, 10)), None, 2, (10,), 1)


def test_multibatch_inference():
    """Verify that inputs with batch_size > 1 are handled correctly."""
    np.random.seed(42)
    # 2 samples in the batch
    inputs = np.random.rand(2, 32, 32, 1).astype(np.float32)

    model = MockModel(output_classes=2, num_spatial_dims=2)
    roi_size = (16, 16)
    sw_batch_size = 4

    output = sliding_window_inference(
        inputs=inputs,
        model=model,
        num_classes=2,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
    )

    assert output.shape == (2, 32, 32, 2)
    # Ensure the two samples have different (correct) predictions
    # Before the fix, they might have been identical if one was broadcasted
    assert not np.allclose(output[0], output[1])
