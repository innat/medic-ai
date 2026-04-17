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
    def __init__(self, output_classes=2):
        self.output_classes = output_classes
        # We need output_shape attribute to mimic the keras model
        self.output_shape = (None, None, None, None, self.output_classes)

    def predict(self, x, verbose=0):
        # Deterministic mock prediction: just append channel dimension if needed or broadcast
        # x shape: (batch_size, d, h, w, c)
        output_shape = list(x.shape)
        output_shape[-1] = self.output_classes
        
        # We can just return x projected to output_classes to ensure deterministic test
        # Let's say we just add 1 and broadcast to output_classes
        res = np.ones(output_shape, dtype=np.float32)
        res = res * np.mean(x, axis=-1, keepdims=True) + 1.0
        return res


@pytest.mark.parametrize("overlap", [0.25, 0.5])
@pytest.mark.parametrize("mode", ["constant", "gaussian"])
@pytest.mark.parametrize("sigma_scale", [0.125, 0.25])
def test_modular_sliding_window_inference_equivalence(overlap, mode, sigma_scale):
    # Set up dummy inputs
    np.random.seed(42)
    inputs = np.random.rand(1, 32, 32, 32, 1).astype(np.float32)  # batch size 1
    
    # Model and inference parameters
    model = MockModel(output_classes=2)
    roi_size = (16, 16, 16)
    sw_batch_size = 4
    num_classes = 2

    # Case 0: The original monolithic function
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

    # Case 1: The new wrapper
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

    # Case 2: Step-by-step sequential execution
    patches, info = extract_patches(
        inputs=inputs,
        roi_size=roi_size,
        overlap=overlap,
        mode=mode,
        sigma_scale=sigma_scale,
    )
    predictions, importance_map_resized = predict_patches(
        patches=patches,
        model=model,
        sw_batch_size=sw_batch_size,
        roi_size=roi_size,
        importance_map=info["importance_map"],
    )
    output_case_2 = merge_patches(
        predictions=predictions,
        info=info,
        importance_map_resized=importance_map_resized,
        num_classes=num_classes,
    )

    # Assert completely identical outputs across all three routes
    np.testing.assert_allclose(output_case_0, output_case_1, rtol=1e-5, atol=1e-5, err_msg="Wrapper output does not match Case 0!")
    np.testing.assert_allclose(output_case_0, output_case_2, rtol=1e-5, atol=1e-5, err_msg="Component pipeline output does not match Case 0!")
