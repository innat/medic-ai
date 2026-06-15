# FAQ

## Which Keras backend should I use?

Use the backend that best matches your training and deployment environment.
`medicai` is designed for Keras 3 workflows across the TensorFlow, PyTorch,
and JAX backends.

In practice:

- Choose `tensorflow` if your pipeline is centered around `tf.data`,
  TensorFlow tooling, or TPU workflows.
- Choose `torch` if your team primarily works in the PyTorch ecosystem.
- Choose `jax` if you prefer JAX-native training and compilation workflows.

One important detail is that the `medicai.transforms` API is currently
implemented with TensorFlow ops. This is intentional so the same transform
layer can integrate with `tf.data`, Python generators, and
`keras.utils.PyDataset`, while still remaining usable in multi-backend Keras
training workflows. This design also reflects a current Keras limitation tracked
in [keras-team/keras#21785](https://github.com/keras-team/keras/issues/21785).

If you want a fully backend-specific data pipeline, you can also mix `medicai`
models with ecosystem-native tooling such as
[TorchIO](https://github.com/TorchIO-project/torchio) for PyTorch-oriented
medical imaging workflows.

## Does medicai support 2D and 3D medical images?

Yes. `medicai` is designed for both 2D and 3D medical imaging workflows.

In general:

- Use input shapes like `(height, width, channels)` for 2D images.
- Use input shapes like `(depth, height, width, channels)` for 3D volumes.

Many model builders automatically construct either a 2D or 3D variant based on
the input shape you provide. This makes it easier to move between slice-based
and volumetric workflows without learning a completely separate API.

That said, support still depends on the specific model or transform. If you are
unsure, check the corresponding model guide or API reference page for that
component.

## Does medicai provide ImageNet pre-trained weights for models?

No, not at the moment.

The library is currently focused on medical imaging workflows, so pre-trained
ImageNet weights have not been a primary focus yet. In some cases they may offer
only limited gains for highly specialized medical domains, but that can vary by
task and dataset.

This may change in the future if there is enough community demand or strong
evidence that pre-trained weights are consistently useful in real-world
`medicai` workflows.


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
