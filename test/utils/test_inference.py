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

    def __init__(self, output_classes=2):
        self.output_classes = output_classes
        # output_shape is only used by the old function's fallback:
        #   num_classes = num_classes or model.output_shape[-1]
        self.output_shape = (None, None, None, None, self.output_classes)

    def predict(self, x, verbose=0):
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

    model = MockModel(output_classes=2)
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

    model = MockModel(output_classes=2)
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
