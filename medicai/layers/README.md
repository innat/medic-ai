# Pooling

## Adaptive Pooling

The AdaptivePooling implementation is cross-platform and device-agnostic, working reliably on ($\text{tf}$), ($\text{torch}$), and $\text{jax}$ across **GPUs** and **TPUs**. Its output is numerically identical to `torch.nn.AdaptiveAvgPool` (within standard tolerances of $1\text{e-}6$).

```python
import numpy as np
import torch
from torch import nn

from medicai.layers import (
    AdaptiveAveragePooling2D,
    AdaptiveAveragePooling3D,
    AdaptiveMaxPooling2D,
    AdaptiveMaxPooling3D,
)


def compare_outputs(
    keras_output,
    torch_output,
    transpose_axes,
    atol: float = 1e-6,
    rtol: float = 1e-6,
):
    # 1. Prepare PyTorch output: detach, convert to NumPy, and transpose
    torch_np = torch_output.detach().numpy().transpose(*transpose_axes)
    # 2. Perform the comparison (raises error on failure)
    np.testing.assert_allclose(keras_output, torch_np, atol=atol, rtol=rtol)
```
```python
def run_test_2d(pool_size=(5, 7)):
    # Input (1, 64, 8, 9)
    torch_input = torch.randn(1, 64, 8, 9)
    # Keras Input conversion (0, 2, 3, 1) -> (1, 8, 9, 64)
    keras_input = ops.convert_to_numpy(torch_input).transpose(0, 2, 3, 1)

    # Max Pool Test
    torch_out_max = nn.AdaptiveMaxPool2d(pool_size)(torch_input)
    keras_out_max = AdaptiveMaxPooling2D(output_size=pool_size)(keras_input)
    compare_outputs(keras_out_max, torch_out_max, transpose_axes=(0, 2, 3, 1))

    # Avg Pool Test
    torch_out_avg = nn.AdaptiveAvgPool2d(pool_size)(torch_input)
    keras_out_avg = AdaptiveAveragePooling2D(output_size=pool_size)(keras_input)
    compare_outputs(keras_out_avg, torch_out_avg, transpose_axes=(0, 2, 3, 1))
    print(f"2D Tests (Max & Avg) passed for pool size {pool_size}.")
```
```python
def run_test_3d(pool_size=(5, 7, 9)):
    # Input (1, 64, 8, 9, 10)
    torch_input = torch.randn(1, 64, 8, 9, 10)
    # Keras Input conversion (0, 2, 3, 4, 1) -> (1, 8, 9, 10, 64)
    keras_input = ops.convert_to_numpy(torch_input).transpose(0, 2, 3, 4, 1)

    # Max Pool Test
    torch_out_max = nn.AdaptiveMaxPool3d(pool_size)(torch_input)
    keras_out_max = AdaptiveMaxPooling3D(output_size=pool_size)(keras_input)
    compare_outputs(keras_out_max, torch_out_max, transpose_axes=(0, 2, 3, 4, 1))

    # Avg Pool Test
    torch_out_avg = nn.AdaptiveAvgPool3d(pool_size)(torch_input)
    keras_out_avg = AdaptiveAveragePooling3D(output_size=pool_size)(keras_input)
    compare_outputs(keras_out_avg, torch_out_avg, transpose_axes=(0, 2, 3, 4, 1))
    print(f"3D Tests (Max & Avg) passed for pool size {pool_size}.")
```
```python
run_test_2d()
run_test_3d()
# 2D Tests (Max & Avg) passed for pool size (5, 7).
# 3D Tests (Max & Avg) passed for pool size (5, 7, 9).
```