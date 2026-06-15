# Installation

You can install `medicai` either from PyPI or directly from the GitHub source repository.

## Install from PyPI

```bash
pip install medicai
```

This installs the core package and its direct Python dependencies, including `keras`,
but it does not install a backend runtime such as TensorFlow, PyTorch, or JAX.

## Install from GitHub Source

```bash
pip install git+https://github.com/innat/medic-ai.git
```

## Backend Framework Support

`medicai` is built on top of **Keras 3** and supports multi-backend workflows across

- `tensorflow`
- `torch`
- `jax`

The package itself does not automatically install any backend framework. You must install your preferred backend separately based on your workflow and hardware setup. For GPU acceleration, make sure:

- CUDA and related GPU drivers are properly installed
- Your selected backend has GPU support enabled
- The framework can successfully detect your GPU or TPU device

Optional extras are available for local workflows:

```bash
pip install "medicai[docs]"
pip install "medicai[test]"
pip install "medicai[dev]"
```

- `docs` installs the documentation toolchain.
- `test` installs the testing toolchain.
- `dev` installs docs, tests, and formatting tools.

## Verifying GPU Access

After installing your preferred backend, you can verify GPU availability using the following examples.

**TensorFlow**

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

If TensorFlow detects your GPU correctly, it will return a list of available GPU devices.

**PyTorch**

```python
import torch
print(torch.cuda.is_available()) 
print(torch.cuda.get_device_name(0))
```

If GPU spport is enabled correctly, you should see cuda availabitly flag is `true` and device name.

**JAX**

```python
import jax 
print(jax.devices())
```

If GPU support is enabled correctly, you should see GPU devices listed instead of only CPU devices.

## Setting the Keras Backend

The backend must be configured before importing **keras**, and it cannot be changed after the package has been imported.

```python
import os 

# "tensorflow" or "torch", "jax" 
os.environ["KERAS_BACKEND"] = "tensorflow" 

import keras
keras.distribution.list_devices()
```

Verify the packages:

```python
import keras
import medicai

print(keras.version())
print(medicai.version())
print(keras.config.backend())
```
