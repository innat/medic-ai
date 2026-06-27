import numpy as np
import pytest

from medicai.utils.inference import SlidingWindowInference, sliding_window_inference


class DummySegmentationModel:
    """Simple spatially preserving test model for sliding-window inference."""

    def __init__(self, num_classes: int):
        self.output_shape = (None, None, None, num_classes)
        self.num_classes = num_classes

    def predict(self, patches: np.ndarray, verbose: int = 0) -> np.ndarray:
        channels = []
        for class_index in range(self.num_classes):
            channels.append(patches[..., :1] + float(class_index))
        return np.concatenate(channels, axis=-1).astype(np.float32)


class RecordingSegmentationModel(DummySegmentationModel):
    """Test model that records the batch dimension seen by predict()."""

    def __init__(self, num_classes: int):
        super().__init__(num_classes=num_classes)
        self.seen_batch_sizes = []

    def predict(self, patches: np.ndarray, verbose: int = 0) -> np.ndarray:
        self.seen_batch_sizes.append(int(patches.shape[0]))
        return super().predict(patches, verbose=verbose)


@pytest.mark.unit
def test_sliding_window_inference_matches_full_prediction_for_large_2d_input():
    model = DummySegmentationModel(num_classes=3)
    inputs = np.random.randn(1, 32, 48, 1).astype(np.float32)

    output = sliding_window_inference(
        x=inputs,
        model=model,
        num_classes=3,
        roi_size=(16, 20),
        sw_batch_size=3,
        overlap=0.25,
    )

    expected = model.predict(inputs, verbose=0)
    np.testing.assert_allclose(output, expected, atol=1e-6)


@pytest.mark.unit
def test_sliding_window_inference_pads_and_crops_3d_input_smaller_than_roi():
    model = DummySegmentationModel(num_classes=2)
    inputs = np.random.randn(1, 8, 10, 12, 1).astype(np.float32)

    output = sliding_window_inference(
        x=inputs,
        model=model,
        num_classes=2,
        roi_size=(12, 14, 16),
        sw_batch_size=2,
        overlap=0.5,
    )

    expected = model.predict(inputs, verbose=0)
    assert output.shape == (1, 8, 10, 12, 2)
    np.testing.assert_allclose(output, expected, atol=1e-6)


@pytest.mark.unit
def test_sliding_window_inference_handles_multi_item_batches():
    model = DummySegmentationModel(num_classes=2)
    inputs = np.random.randn(2, 18, 22, 1).astype(np.float32)

    output = sliding_window_inference(
        x=inputs,
        model=model,
        num_classes=None,
        roi_size=(9, 11),
        sw_batch_size=4,
        overlap=0.5,
    )

    expected = model.predict(inputs, verbose=0)
    assert output.shape == (2, 18, 22, 2)
    np.testing.assert_allclose(output, expected, atol=1e-6)


@pytest.mark.unit
def test_sliding_window_inference_preserves_sample_identity_across_five_volumes():
    model = DummySegmentationModel(num_classes=2)
    base = np.zeros((5, 16, 18, 20, 1), dtype=np.float32)
    for sample_index in range(base.shape[0]):
        base[sample_index, ..., 0] = float(sample_index + 1)

    output = sliding_window_inference(
        x=base,
        model=model,
        num_classes=2,
        roi_size=(10, 12, 14),
        sw_batch_size=2,
        overlap=0.25,
    )

    expected = model.predict(base, verbose=0)
    assert output.shape == (5, 16, 18, 20, 2)
    np.testing.assert_allclose(output, expected, atol=1e-6)
    for sample_index in range(base.shape[0]):
        np.testing.assert_allclose(output[sample_index, ..., 0], sample_index + 1, atol=1e-6)
        np.testing.assert_allclose(output[sample_index, ..., 1], sample_index + 2, atol=1e-6)


@pytest.mark.unit
def test_sliding_window_inference_skips_padding_for_single_patch_batch():
    model = RecordingSegmentationModel(num_classes=2)
    x = np.random.randn(1, 8, 10, 12, 1).astype(np.float32)

    output = sliding_window_inference(
        x=x,
        model=model,
        num_classes=2,
        roi_size=(12, 14, 16),
        sw_batch_size=8,
        overlap=0.25,
    )

    assert output.shape == (1, 8, 10, 12, 2)
    assert model.seen_batch_sizes == [1]


@pytest.mark.unit
def test_sliding_window_inference_class_wrapper_preserves_behavior():
    model = DummySegmentationModel(num_classes=2)
    inputs = np.random.randn(1, 20, 24, 1).astype(np.float32)
    inferencer = SlidingWindowInference(
        model=model,
        num_classes=2,
        roi_size=(10, 12),
        sw_batch_size=2,
        overlap=0.25,
    )

    output = inferencer(inputs)
    expected = model.predict(inputs, verbose=0)
    np.testing.assert_allclose(output, expected, atol=1e-6)


@pytest.mark.unit
def test_sliding_window_inference_accepts_scalar_roi_weight_map():
    model = DummySegmentationModel(num_classes=2)
    inputs = np.random.randn(1, 16, 20, 1).astype(np.float32)

    output = sliding_window_inference(
        x=inputs,
        model=model,
        num_classes=2,
        roi_size=(8, 10),
        sw_batch_size=2,
        overlap=0.25,
        roi_weight_map=0.8,
    )

    expected = model.predict(inputs, verbose=0)
    np.testing.assert_allclose(output, expected, atol=1e-6)


@pytest.mark.unit
def test_sliding_window_inference_accepts_spatial_only_roi_weight_map():
    model = DummySegmentationModel(num_classes=2)
    inputs = np.random.randn(1, 16, 20, 1).astype(np.float32)
    roi_weight_map = np.ones((8, 10), dtype=np.float32)

    output = sliding_window_inference(
        x=inputs,
        model=model,
        num_classes=2,
        roi_size=(8, 10),
        sw_batch_size=2,
        overlap=0.25,
        roi_weight_map=roi_weight_map,
    )

    expected = model.predict(inputs, verbose=0)
    np.testing.assert_allclose(output, expected, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("padding_mode", ["replicate", "circular"])
def test_sliding_window_inference_supports_documented_padding_aliases(padding_mode):
    model = DummySegmentationModel(num_classes=2)
    inputs = np.random.randn(1, 6, 7, 1).astype(np.float32)

    output = sliding_window_inference(
        x=inputs,
        model=model,
        num_classes=2,
        roi_size=(10, 11),
        sw_batch_size=1,
        overlap=0.25,
        padding_mode=padding_mode,
    )

    assert output.shape == (1, 6, 7, 2)
