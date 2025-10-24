

<img src="assets/logo.jpg" width="500"/>


[![Palestine](https://img.shields.io/badge/Free-Palestine-white?labelColor=green)](https://twitter.com/search?q=%23FreePalestine&src=typed_query) 

![Static Badge](https://img.shields.io/badge/keras-3.9.0-darkred?style=flat) ![Static Badge](https://img.shields.io/badge/tensorflow-2.19.0-orange?style=flat) ![Static Badge](https://img.shields.io/badge/torch-2.6.0-red?style=flat) ![Static Badge](https://img.shields.io/badge/jax-0.4.23-%233399ff)

**Medic-AI** is a [Keras](https://keras.io/keras_3/) based library designed for medical 2D and 3D image analysis using machine learning techniques. Its core strengths include:

- **Backend Agnostic:** Compatible with `tensorflow`, `torch`, and `jax`.
- **User-Friendly API:** High-level interface for transformations and model creation of both 2D and 3D.
- **Scalable Execution:** Supports training and inference on **single/multi-GPU** and **TPU-VM** setups.
- **Essential Components:** Includes standard medical specific metrics and losses, such as Dice.
- **Optimized 3D Inference:** Offers an efficient sliding-window method and callback for volumetric data.


# 📋 Table of Contents
1. [Installation](#-installation)
2. [Features](#-features)
3. [Guides](#-guides)
4. [Documentation](#-documentation)
5. [Acknowledgements](#-acknowledgements)
6. [Citation](#-citation)


# 🛠 Installation

PyPI version:

```bash
!pip install medicai
```

Installing from source GitHub: (**recommended**)

```bash
!pip install git+https://github.com/innat/medic-ai.git
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

The available `model/encoder` can be listed down, showing below.

```python
import medicai
medicai.models.list_models()

                   Model Registry Catalog
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Segmentor        ┃ Backbone Family ┃ Variants            ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ • attention_unet │ convnext        │ • convnext_base     │
│ • unet           │                 │ • convnext_large    │
│ • unet_plus_plus │                 │ • convnext_small    │
│                  │                 │ • convnext_tiny     │
│                  │                 │ • convnext_xlarge   │
│                  │                 │ • convnextv2_atto   │
│                  │                 │ • convnextv2_base   │
│                  │                 │ • convnextv2_femto  │
│                  │                 │ • convnextv2_huge   │
│                  │                 │ • convnextv2_large  │
│                  │                 │ • convnextv2_nano   │
│                  │                 │ • convnextv2_pico   │
│                  │                 │ • convnextv2_small  │
│                  │                 │ • convnextv2_tiny   │
├──────────────────┼─────────────────┼─────────────────────┤
│ • attention_unet │ densenet        │ • densenet121       │
│ • trans_unet     │                 │ • densenet169       │
│ • unet           │                 │ • densenet201       │
│ • unet_plus_plus │                 │                     │
├──────────────────┼─────────────────┼─────────────────────┤
│ • attention_unet │ efficientnet    │ • efficientnet_b0   │
│ • unet           │                 │ • efficientnet_b1   │
│ • unet_plus_plus │                 │ • efficientnet_b2   │
│                  │                 │ • efficientnet_b3   │
│                  │                 │ • efficientnet_b4   │
│                  │                 │ • efficientnet_b5   │
│                  │                 │ • efficientnet_b6   │
│                  │                 │ • efficientnet_b7   │
│                  │                 │ • efficientnet_b8   │
│                  │                 │ • efficientnet_l2   │
│                  │                 │ • efficientnetv2_b0 │
│                  │                 │ • efficientnetv2_b1 │
│                  │                 │ • efficientnetv2_b2 │
│                  │                 │ • efficientnetv2_b3 │
│                  │                 │ • efficientnetv2_l  │
│                  │                 │ • efficientnetv2_m  │
│                  │                 │ • efficientnetv2_s  │
├──────────────────┼─────────────────┼─────────────────────┤
│ • segformer      │ mit             │ • mit_b0            │
│                  │                 │ • mit_b1            │
│                  │                 │ • mit_b2            │
│                  │                 │ • mit_b3            │
│                  │                 │ • mit_b4            │
│                  │                 │ • mit_b5            │
├──────────────────┼─────────────────┼─────────────────────┤
│ • attention_unet │ resnet          │ • resnet101         │
│ • trans_unet     │                 │ • resnet101v2       │
│ • unet           │                 │ • resnet152         │
│ • unet_plus_plus │                 │ • resnet152v2       │
│                  │                 │ • resnet18          │
│                  │                 │ • resnet200vd       │
│                  │                 │ • resnet34          │
│                  │                 │ • resnet50          │
│                  │                 │ • resnet50v2        │
│                  │                 │ • resnet50vd        │
├──────────────────┼─────────────────┼─────────────────────┤
│ • swin_unetr     │ swin            │ • swin_base         │
│                  │                 │ • swin_base_v2      │
│                  │                 │ • swin_small        │
│                  │                 │ • swin_small_v2     │
│                  │                 │ • swin_tiny         │
│                  │                 │ • swin_tiny_v2      │
├──────────────────┼─────────────────┼─────────────────────┤
│ • unetr          │ vit             │ • vit_base          │
│                  │                 │ • vit_huge          │
│                  │                 │ • vit_large         │
.....
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
| [**UNETR**](medicai/models/unetr/README.md) | 2D, 3D | Segmentation | Transformer |
| [**SwinUNETR**](medicai/models/swin/README.md) | 2D, 3D | Segmentation | Transformer |
| [**SwinUNETR-V2**](medicai/models/swin/README.md) | 2D, 3D | Segmentation | Transformer |
| [**TransUNet**](medicai/models/transunet/README.md) | 2D, 3D | Segmentation | Transformer |
| [**SegFormer**](medicai/models/segformer/README.md) | 2D, 3D | Segmentation | Transformer |

**Available Transformation**: The following preprocessing and transformation methods are supported for volumetric data. The following layers are implemented with **TensorFlow** operations. It can be used in the `tf.data` API or a Python data generator and is fully compatible with multiple backends, `tf`, `torch`, `jax` in training and inference, supporting both GPUs and TPUs.

```bash
CropForeground
NormalizeIntensity
Orientation
RandCropByPosNegLabel
RandFlip
RandRotate90
RandShiftIntensity
RandSpatialCrop
Resize
ScaleIntensityRange
Spacing
```


# 💡 Guides

**Segmentation**: Available guides for 3D segmentation task.

| Task | GitHub | Kaggle | View |
|----------|----------|----------|----------|
| Covid-19  | <a target="_blank" href="notebooks/covid19.ct.segment.ipynb"><img src="https://img.shields.io/badge/GitHub-View%20source-lightgrey" /></a>     | <a target="_blank" href="https://www.kaggle.com/code/ipythonx/medicai-covid-19-3d-image-segmentation/notebook"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>     | <img src="assets/covid.gif" width="200"/>    |
| BTCV  | <a target="_blank" href="notebooks/btcv.segment.ipynb"><img src="https://img.shields.io/badge/GitHub-View%20source-lightgrey" /></a>    | <a target="_blank" href="https://www.kaggle.com/code/ipythonx/medicai-3d-btcv-segmentation-in-keras/"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>    | n/a     |
| BraTS  | <a target="_blank" href="notebooks/brats.multi-gpu.segment.ipynb"><img src="https://img.shields.io/badge/GitHub-View%20source-lightgrey" /></a>     | <a target="_blank" href="https://www.kaggle.com/code/ipythonx/3d-brats-segmentation-in-keras-multi-gpu/"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>    | n/a     |
| Spleen | <a target="_blank" href="notebooks/spleen.segment.ipynb"><img src="https://img.shields.io/badge/GitHub-View%20source-lightgrey" /></a>     | <a target="_blank" href="https://www.kaggle.com/code/ipythonx/medicai-spleen-3d-segmentation-in-keras"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>     | <img src="assets/spleen.gif" width="200">  |

**Classification**: Available guides for 3D classification task.

| Task (Classification) | GitHub | Kaggle |
|----------|----------|----------|
| Covid-19   | <a target="_blank" href="notebooks/covid19.ct.classification.ipynb"><img src="https://img.shields.io/badge/GitHub-View%20source-lightgrey" /></a>       | <a target="_blank" href="https://www.kaggle.com/code/ipythonx/medicai-3d-image-classification"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>     |


# 📚 Documentation

To learn more about **model**, **transformation**, and **training**, please visit official documentation: [`medicai/docs`](https://innat.github.io/medic-ai/)

# 🤝 Contributing

Please check the contribution guide [here](CONTRIBUTION.md).


# 🙏 Acknowledgements

This project is greatly inspired by [MONAI](https://monai.io/).

# 📝 Citation

If you use `medicai` in your research or educational purposes, please cite it using the metadata from our [`CITATION.cff`](https://github.com/innat/medic-ai/blob/main/CITATION.cff) file.
