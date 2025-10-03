

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

# Quick Start

```python
from medicai.models import SwinUNETR, UNet
from medicai.models import SwinTiny, SwinTinyV2
from medicai.models import SwinBackbone, SwinBackboneV2

# Build 3D model.
model = SwinUNETR(encoder_name='swin_tiny_v2', input_shape=(96,96,96,1))
model = UNet(encoder_name='densenet121', input_shape=(96,96,96,1))

# Build 2D model.
model = SwinUNETR(encoder_name='swin_tiny_v2', input_shape=(96,96,1))
model = UNet(encoder_name='swin_tiny_v2', input_shape=(96,96,1))
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
| **DenseNet** | 2D, 3D | Classification | CNN |
| **ResNet (V1/V2)** | 2D, 3D | Classification | CNN |
| **ViT** | 2D, 3D | Classification | Transformer |
| **MiT** | 2D, 3D | Classification | Transformer |
| **Swin Transformer (V1/V2)** | 2D, 3D | Classification | Transformer |
| **UNet** | 2D, 3D | Segmentation | CNN |
| **UNETR** | 2D, 3D | Segmentation | Transformer |
| **SwinUNETR** | 2D, 3D | Segmentation | Transformer |
| **TransUNet** | 2D, 3D | Segmentation | Transformer |
| **SegFormer** | 2D, 3D | Segmentation | Transformer |

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

Please refer to the current [roadmap](https://github.com/innat/medic-ai/wiki/Roadmap) for an overview of the project. Feel free to explore anything that interests you. If you have suggestions or ideas, I’d appreciate it if you could open a [GitHub issue](https://github.com/innat/medic-ai/issues/new/choose) so we can discuss them further.

1. Install `medicai` from soruce:

```bash
!git clone https://github.com/innat/medic-ai
%cd medic-ai
!pip install keras -qU
!pip install -e .
%cd ..
```

Add your contribution and implement relevant test code.

2. Run test code as:

```
python -m pytest test/

# or, only one your new_method
python -m pytest -k new_method
```

# 🙏 Acknowledgements

This project is greatly inspired by [MONAI](https://monai.io/).

# 📝 Citation

If you use `medicai` in your research or educational purposes, please cite it using the metadata from our [`CITATION.cff`](https://github.com/innat/medic-ai/blob/main/CITATION.cff) file.
