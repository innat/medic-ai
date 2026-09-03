<img src="docs/_static/medicai-wordmark-wide.svg" alt="medicai" width="360">

[![Palestine](https://img.shields.io/badge/Free-Palestine-white?labelColor=green)](https://twitter.com/search?q=%23FreePalestine&src=typed_query) 

![Static Badge](https://img.shields.io/badge/keras-3.15.0-darkred?style=flat) ![Static Badge](https://img.shields.io/badge/tensorflow-2.21.0-orange?style=flat) ![Static Badge](https://img.shields.io/badge/torch-2.5.1-red?style=flat) ![Static Badge](https://img.shields.io/badge/jax-0.1-blue?style=flat)

**Medic-AI** is a [Keras](https://keras.io/keras_3/) based library designed for medical 2D and 3D image analysis using machine learning techniques. Its core strengths include:

- **Backend Agnostic:** Compatible with `tensorflow`, `torch`, and `jax`.
- **User-Friendly API:** High-level interface for `75+` classification models, `11+` segmentation models, and `20+` medical preprocessing transforms across 2D and 3D.
- **Flexible Transformations:** Provides backend-agnostic preprocessing and augmentation for 2D and 3D images, with both **single and batch** support, synchronized image-label processing, and CPU/GPU execution where supported.
- **Scalable Execution:** Supports training and inference on **single/multi-GPU** and **TPU-VM** setups.
- **Essential Components:** Includes standard medical specific metrics and losses. Support **GradCAM** for segmentation and classification on both 2D and 3D input including large volume of medical inputs.
- **Optimized 3D Inference:** Offers an efficient sliding-window method and callback for volumetric data.


# 📋 Table of Contents
1. [Installation](#-installation)
2. [Features](#-features)
3. [Documentation](#-documentation)
4. [Acknowledgements](#-acknowledgements)
5. [Citation](#-citation)


# 🛠 Installation

PyPI version:

```bash
pip install medicai
```

This installs `medicai` and `keras`, but leaves backend runtime selection to you.
Install `tensorflow`, `torch`, or `jax` separately based on your workflow.

Installing from source GitHub: (**recommended**)

```bash
pip install git+https://github.com/innat/medic-ai.git
```

Using `uv` for local development:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .[dev]
```

Optional extras:

```bash
pip install "medicai[docs]"
pip install "medicai[test]"
pip install "medicai[dev]"
```

# Quick Overview

For details end-to-end training workflow, please check the [code-example](https://medicai.readthedocs.io/en/latest/guides/example.html) section.

```python
from medicai.models import SwinUNETR, UNet
from medicai.models import SwinTiny, SwinTinyV2
from medicai.models import SwinBackbone, SwinBackboneV2

# Build 3D model.
model = SwinUNETR(
    encoder_name='swin_tiny_v2', input_shape=(96,96,96,1)
)
model = UNet(
    encoder_name='densenet121', input_shape=(96,96,96,1)
)

# Build 2D model.
model = SwinUNETR(
    encoder_name='swin_tiny_v2', input_shape=(96,96,1)
)
model = UNet(
    encoder_name='densenet121', input_shape=(96,96,1)
)
```
```python
# Build with pre-built encoder.
encoder = SwinTiny(
    input_shape=(96,96,96,1),
    patch_size=2, 
    downsampling_strategy='swin_unetr_like'
)
model = SwinUNETR(encoder=encoder)

# Build with custom encoder.
custom_encoder = SwinBackboneV2(
    input_shape=(64, 128, 128, 1),
    embed_dim=48,
    window_size=8,
    patch_size=2,
    downsampling_strategy='swin_unetr_like'
)
model = SwinUNETR(encoder=custom_encoder)
```

# 📊 Features

**Supported Dataloaders**: Choose the right dataloaders for the targeted Keras backend.

| Keras backend | PyGrain | `torch.utils.data` | `tf.data` | `keras.utils.PyDataset` |
| :--- | :---: | :---: | :---: | :---: |
| TensorFlow | ✓ | ✗ | ✓ | ✓ |
| Torch | ✓ | ✓ | ✗ | ✓ |
| JAX | ✓ | ✗ | ✗ | ✓ |

> **Cross-backend pipelines:** If the same end-to-end pipeline must run with
> every Keras backend, choose **PyGrain** or `keras.utils.PyDataset`. PyGrain is
> highly recommended for its efficient parallel data loading, worker support,
> and backend-neutral record pipeline. `keras.utils.PyDataset` is a simpler
> choice for custom Python datasets.

**Available Models** : The following table lists the currently supported models along with their supported input modalities, primary tasks, and underlying architecture type.  The model inputs can be either **3D** `(depth × height × width × channel)` or **2D** `(height × width × channel)`.

| Model | Supported Modalities | Primary Task | Architecture Type |
| :--- | :--- | :--- | :--- |
| [**DenseNet**](medicai/models/densenet/README.md) | 2D, 3D | Classification | CNN |
| [**ResNet-V1,V2**](medicai/models/resnet/README.md) | 2D, 3D | Classification | CNN |
| [**ResNeXt**](medicai/models/resnet/README.md) | 2D, 3D | Classification | CNN |
| [**SE-ResNet**](medicai/models/senet/README.md) | 2D, 3D | Classification | CNN |
| [**SE-ResNeXt**](medicai/models/senet/README.md) | 2D, 3D | Classification | CNN |
| [**Xception**](medicai/models/xception/README.md) | 2D, 3D | Classification | CNN |
| [**EfficientNet-V1,V2**](medicai/models/efficientnet/README.md) | 2D, 3D | Classification | CNN |
| [**ConvNeXt-V1,V2**](medicai/models/convnext/README.md) | 2D, 3D | Classification | CNN |
| [**ViT**](medicai/models/vit/README.md) | 2D, 3D | Classification | Transformer |
| [**MiT**](medicai/models/mit/README.md) | 2D, 3D | Classification | Transformer |
| [**Swin Transformer-V1,V2**](medicai/models/swin/README.md) | 2D, 3D | Classification | Transformer |
| [**UNet**](medicai/models/unet/README.md) | 2D, 3D | Segmentation | CNN |
| [**UNet++**](medicai/models/unet_plus_plus/README.md) | 2D, 3D | Segmentation | CNN |
| [**AttentionUNet**](medicai/models/unet/README.md) | 2D, 3D | Segmentation | CNN |
| [**DeepLabV3Plus**](medicai/models/deeplabv3plus/README.md) | 2D, 3D | Segmentation | CNN |
| [**UPerNet**](medicai/models/upernet/README.md) | 2D, 3D | Segmentation | CNN |
| [**UNETR**](medicai/models/unetr/README.md) | 2D, 3D | Segmentation | Transformer |
| [**UNETR++**](medicai/models/unetr_plus_plus/README.md) | 2D, 3D | Segmentation | Transformer |
| [**SwinUNETR**](medicai/models/swin/README.md) | 2D, 3D | Segmentation | Transformer |
| [**SwinUNETR-V2**](medicai/models/swin/README.md) | 2D, 3D | Segmentation | Transformer |
| [**TransUNet**](medicai/models/transunet/README.md) | 2D, 3D | Segmentation | Transformer |
| [**SegFormer**](medicai/models/segformer/README.md) | 2D, 3D | Segmentation | Transformer |

**Available Transformations**: The following preprocessing and augmentation
transforms are implemented with backend-native `keras.ops` and support common
2D and 3D channel-last layouts in both forward and inverse pass. Transforms declare whether they 
accept sample inputs (`HWC`, `DHWC`), batch inputs (`BHWC`, `BDHWC`), or both. Most transforms
can run on either CPU or GPU, depending on the active backend and execution
context. Many transforms are also compatible with XLA compilation; any
backend-specific XLA limitations are documented by the individual transform.

| Transformation | Layout | GPU | GPU (XLA/compiled) |
| :--- | :--- | :---: | :---: |
| NormalizeIntensity | `HWC`, `DHWC`, `BHWC`, `BDHWC` | Yes | Yes |
| ScaleIntensityRange | `HWC`, `DHWC`, `BHWC`, `BDHWC` | Yes | Yes |
| ShiftIntensity | `HWC`, `DHWC`, `BHWC`, `BDHWC` | Yes | Yes |
| SignalFillEmpty | `HWC`, `DHWC`, `BHWC`, `BDHWC` | Yes | Yes |
| Flip | `HWC`, `DHWC`, `BHWC`, `BDHWC` | Yes | Yes |
| SpatialCrop | `HWC`, `DHWC`, `BHWC`, `BDHWC` | Yes | Limited |
| Resize | `HWC`, `DHWC`, `BHWC`, `BDHWC` | Yes | Limited |
| Rotate90 | `HWC`, `DHWC`, `BHWC`, `BDHWC` | Yes | Yes |
| RandomCropByPosNegLabel | `HWC`, `DHWC`, `BHWC`, `BDHWC` | Yes | Limited |
| RandomCutOut | `HWC`, `DHWC`, `BHWC`, `BDHWC` | Yes | Yes |
| RandomRotate | `HWC`, `DHWC`, `BHWC`, `BDHWC` | Yes | Limited |
| RandomRotate90 | `HWC`, `DHWC`, `BHWC`, `BDHWC` | Yes | Yes |
| RandomFlip | `HWC`, `DHWC`, `BHWC`, `BDHWC` | Yes | Yes |
| RandomShiftIntensity | `HWC`, `DHWC`, `BHWC`, `BDHWC` | Yes | Yes |
| RandomSpatialCrop | `HWC`, `DHWC`, `BHWC`, `BDHWC` | Yes | Limited |
| CropForeground | `HWC`, `DHWC` | No | No |
| Orientation | `DHWC` | No | No |
| Spacing | `DHWC` | No | No |

> `Limited` means compiled execution depends on the active backend and runtime
> configuration (i.e., `jit_compile : bool`). `No` indicates that the current  
> implementation is not included
> in the compiled GPU path. The table describes supported execution patterns;
> refer to the transform docstrings and [Transform Benchmarks](benchmarks/README.md)
> for backend-specific limitations and measurements.

**Transform Benchmark Snapshot**: The following compact tables show forward
median execution time on a Tesla T4 GPU. They use the benchmark artifacts with
50 measured iterations and 10 warm-up iterations. The five transforms below
are representative lightweight operations shared across all three backends;
the fastest backend in each row is shown in **bold**. For the complete CPU,
GPU, and compiled results, see [Transform Benchmarks](benchmarks/README.md).

### 2D GPU (`BHWC`)

Batch size is 4 and the channel count is 1.

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | ShiftIntensity | **0.66** | 0.80 | 0.97 |
| BHWC | (4, 224, 224, 1) | ScaleIntensityRange | 1.32 | **0.95** | 1.80 |
| BHWC | (4, 224, 224, 1) | Flip | 1.24 | **0.84** | 1.46 |
| BHWC | (4, 224, 224, 1) | SignalFillEmpty | 1.68 | **1.15** | 2.38 |
| BHWC | (4, 224, 224, 1) | Rotate90 | 3.56 | **0.90** | 2.82 |
| BHWC | (4, 512, 512, 1) | ShiftIntensity | **2.04** | 2.93 | 2.44 |
| BHWC | (4, 512, 512, 1) | ScaleIntensityRange | **2.45** | 3.19 | 3.33 |
| BHWC | (4, 512, 512, 1) | Flip | **2.94** | 3.12 | 3.85 |
| BHWC | (4, 512, 512, 1) | SignalFillEmpty | **2.85** | 3.12 | 3.77 |
| BHWC | (4, 512, 512, 1) | Rotate90 | 7.66 | **3.12** | 4.85 |

### 3D GPU (`BDHWC`)

Batch size is 1 and the channel count is 1.

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BDHWC | (1, 96, 96, 96, 1) | ShiftIntensity | **1.77** | 2.75 | 2.03 |
| BDHWC | (1, 96, 96, 96, 1) | ScaleIntensityRange | **2.10** | 2.59 | 2.84 |
| BDHWC | (1, 96, 96, 96, 1) | Flip | **2.37** | 2.88 | 3.36 |
| BDHWC | (1, 96, 96, 96, 1) | SignalFillEmpty | **2.81** | 3.19 | 3.74 |
| BDHWC | (1, 96, 96, 96, 1) | Rotate90 | 6.71 | **2.95** | 4.59 |
| BDHWC | (1, 256, 256, 256, 1) | ShiftIntensity | 104.17 | 157.50 | **76.23** |
| BDHWC | (1, 256, 256, 256, 1) | ScaleIntensityRange | 107.23 | 159.67 | **78.02** |
| BDHWC | (1, 256, 256, 256, 1) | Flip | 127.52 | 159.66 | **106.23** |
| BDHWC | (1, 256, 256, 256, 1) | SignalFillEmpty | 108.10 | 161.55 | **78.67** |
| BDHWC | (1, 256, 256, 256, 1) | Rotate90 | 266.93 | 158.05 | **107.67** |

> **Note**: These values are representative measurements rather than universal
> performance guarantees. Backend versions, device type, memory pressure,
> warm-up policy, and input dtype can change the relative timings. Randomized,
> resampling, metadata-aware, and XLA-compiled behavior is reported separately
> in the detailed benchmark results.

## Documentation

To learn more about **models**, **transformations**, and **training**, please visit the Read the Docs documentation: [`medicai.readthedocs.io`](https://medicai.readthedocs.io/)

## Contributing

Please check the contribution guide [here](CONTRIBUTION.md).


## Acknowledgements

This project is greatly inspired by [MONAI](https://monai.io/) and [NiftyNet](https://github.com/niftk/niftynet).

## Citation

If you use `medicai` in your research or educational purposes, please cite it using the metadata from our `CITATION.cff` file.
