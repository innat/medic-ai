# FAQ

## Which Keras backend should I use?

Use a backend that best matches your training and deployment environment.
`medicai` is designed for Keras 3 workflows across the TensorFlow, PyTorch,
and JAX backends.

The `medicai` API uses backend-native `keras.ops`, so transforms can
run with the selected TensorFlow, PyTorch, or JAX backend. The data-loading API
should match that backend when transforms are executed outside the model.

| Keras backend | PyGrain | `torch.utils.data` | `tf.data` | `keras.utils.PyDataset` |
| :--- | :---: | :---: | :---: | :---: |
| TensorFlow | ✓ | ✗ | ✓ | ✓ |
| Torch | ✓ | ✓ | ✗ | ✓ |
| JAX | ✓ | ✗ | ✗ | ✓ |


## Does medicai support 2D and 3D medical images?

Yes. `medicai` is designed for both 2D and 3D medical imaging workflows with **channel-last** format.

For the `medicai` models API:

- 2D images: `(height, width, channels)`
- 3D volumes: `(depth, height, width, channels)`

For the `medicai` transforms API, the corresponding **channel-last** layouts are:

| Data type | 2D | 3D |
| :--- | :--- | :--- |
| Single sample | `HWC` | `DHWC` |
| Batch | `BHWC` | `BDHWC` |

Most transforms support both sample-level and batch-level layouts. Some
operations are intentionally sample-only because they inspect spatial
content or metadata for one image at a time, including `CropForeground`,
`Orientation`, `Spacing`, and `RandomCropByPosNegLabel`. Check each transform's
documentation for its supported layouts and backend/XLA limitations.

For segmentation, pass aligned image and label tensors through the same
spatial transform using multiple keys. Spatial transforms reuse the same
sampled geometry for those keys, while image and label interpolation can be
configured independently where supported.


## Does medicai provide ImageNet pre-trained weights for models?

No, not at the moment. It will be supported soon for all encoders.


## How do I report an issue?

Open a GitHub issue at
[innat/medic-ai](https://github.com/innat/medic-ai/issues/new/choose) with:

- a minimal reproduction
- your Python, Keras, and backend versions
- the backend you are using (`tensorflow`, `torch`, or `jax`)
- the expected behavior and the actual behavior

If possible, reproduce the issue in a small Colab or Kaggle notebook and share
the link. Those platforms are especially helpful because they provide free GPU
and TPU environments, which makes backend- and accelerator-related bugs much
easier to reproduce.
