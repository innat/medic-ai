<img src="docs/_static/medicai-wordmark-wide.svg" alt="medicai" width="360">

[![Palestine](https://img.shields.io/badge/Free-Palestine-white?labelColor=green)](https://twitter.com/search?q=%23FreePalestine&src=typed_query) 

![Static Badge](https://img.shields.io/badge/keras-3.15.0-darkred?style=flat) ![Static Badge](https://img.shields.io/badge/tensorflow-2.21.0-orange?style=flat) ![Static Badge](https://img.shields.io/badge/torch-2.5.1-red?style=flat) ![Static Badge](https://img.shields.io/badge/jax-0.1-blue?style=flat)

**Medic-AI** is a [Keras](https://keras.io/keras_3/) based library designed for medical 2D and 3D image analysis using machine learning techniques. Its core strengths include:

- **Backend Agnostic:** Compatible with `tensorflow`, `torch`, and `jax`.
- **User-Friendly API:** High-level interface for transformations and model creation of both 2D and 3D.
- **Scalable Execution:** Supports training and inference on **single/multi-GPU** and **TPU-VM** setups.
- **Essential Components:** Includes standard medical specific metrics and losses, such as Dice. Support **GradCAM** for segmentation and classification on both 2D and 3D input.
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

For details end-to-end training workflow, please check the [guide](#-guides) section.

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

**Available Transformation**: The following preprocessing and transformation methods are supported for volumetric data. The following layers are implemented with **TensorFlow** operations. It can be used in the `tf.data` API or a Python data generator and is fully compatible with multiple backends, `tf`, `torch`, `jax` in training and inference, supporting both GPUs and TPUs.

```bash
CropForeground
NormalizeIntensity
Orientation
RandomCropByPosNegLabel
RandomFlip
RandomRotate90
RandomRotate
RandomCutOut
RandomShiftIntensity
RandomSpatialCrop
Resize
ScaleIntensityRange
Spacing
```


# 📚 Documentation

To learn more about **models**, **transformations**, and **training**, please visit the Read the Docs documentation: [`medicai.readthedocs.io`](https://medicai.readthedocs.io/)

# 🤝 Contributing

Please check the contribution guide [here](CONTRIBUTION.md).


# 🙏 Acknowledgements

This project is greatly inspired by [MONAI](https://monai.io/).

# 📝 Citation

If you use `medicai` in your research or educational purposes, please cite it using the metadata from our `CITATION.cff` file.
