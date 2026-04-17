import sys
import os
import numpy as np

# Add the project root to sys.path
# In WSL the path would be /mnt/c/Users/ASUS/Desktop/medic-ai
# But since I'm running wsl python3 verifier.py I can just use relative paths if I'm in the right CWD
sys.path.append(os.getcwd())

from medicai.utils.inference import sliding_window_inference, extract_patches, merge_patches, predict_patches

class MockModel:
    def __init__(self, output_classes=2):
        self.output_classes = output_classes
        self.output_shape = (None, None, None, 2)
    def predict(self, x, verbose=0):
        bs = x.shape[0]
        # Return something based on input to check batching
        return np.ones((bs, x.shape[1], x.shape[2], self.output_classes)) * (np.arange(bs).reshape(-1, 1, 1, 1) + 1)

def test_multibatch():
    print("Testing multi-batch inference...")
    inputs = np.zeros((2, 32, 32, 1))
    model = MockModel(output_classes=2)
    output = sliding_window_inference(inputs, model, 2, (16, 16), 4)
    assert output.shape == (2, 32, 32, 2)
    # Each sample should have different values because of our MockModel logic
    # Sample 0 should be 1.0, Sample 1 should be 2.0 (approximately, depending on overlaps)
    print(f"Sample 0 Mean: {np.mean(output[0])}")
    print(f"Sample 1 Mean: {np.mean(output[1])}")
    assert not np.allclose(output[0], output[1]), "Multi-batch output should not be identical!"
    print("Multi-batch test PASSED.")

def test_metadata():
    print("Testing extract_patches metadata...")
    inputs = np.zeros((1, 10, 10, 1))
    roi_size = (16, 16)
    padded_inputs, info = extract_patches(inputs, roi_size)
    assert padded_inputs.shape == (1, 16, 16, 1)
    assert info["original_image_size"] == (10, 10)
    assert info["padded_image_size"] == (16, 16)
    print("Metadata test PASSED.")

def test_error_wrapping():
    print("Testing error wrapping...")
    def failing_model(x, verbose=0):
        raise ValueError("Simulated predict error")
    
    class Wrapper:
        def predict(self, x, verbose=0):
            return failing_model(x, verbose)
            
    try:
        sliding_window_inference(np.zeros((1, 32, 32, 1)), Wrapper(), 2, (16, 16), 1)
    except RuntimeError as e:
        print(f"Caught expected error: {e}")
        assert "failed during the prediction/merging phase" in str(e)
        print("Error wrapping test PASSED.")
    except Exception as e:
        print(f"Caught UNEXPECTED error type: {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        test_metadata()
        test_multibatch()
        test_error_wrapping()
        print("\nAll verification tests PASSED.")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
