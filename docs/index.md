
<img src="https://i.imgur.com/GvFKwDG.jpeg" width="500"/>

# MedicAI

**Medic-AI** is a [Keras](https://keras.io/keras_3/) based library designed for medical image analysis using machine learning techniques. Its core strengths include:

- **Backend Agnostic:** Compatible with `tensorflow`, `torch`, and `jax` backends.
- **User-Friendly API:** High-level interface for transformations and model creation.
- **Scalable Execution:** Supports training and inference on **single/multi-GPU** and **TPU-VM** setups.
- **Essential Components:** Includes standard metrics and losses, such as Dice.
- **Optimized 3D Inference:** Offers an efficient sliding-window method and callback for volumetric data


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

Installing from source GitHub:

```bash
!pip install git+https://github.com/innat/medic-ai.git
```

# 📊 Features

**Available Models** : The following table lists the currently supported models along with their supported input modalities, primary tasks, and underlying architecture type.  The model inputs can be either **3D** `(depth × height × width × channel)` or **2D** `(height × width × channel)`.

| Model        | Supported Modalities | Primary Task   | Architecture Type         |
| ------------ | -------------------- | -------------- | ------------------------- |
| DenseNet121     | 2D, 3D               | Classification | CNN                       |
| DenseNet169     | 2D, 3D               | Classification | CNN                       |
| DenseNet201     | 2D, 3D               | Classification | CNN                       |
| ViT          | 2D, 3D               | Classification | Transformer               |
 Swin Transformer          | 2D, 3D               | Classification | Transformer               |
| DenseUNet121 | 2D, 3D               | Segmentation   | CNN                       |
| DenseUNet169 | 2D, 3D               | Segmentation   | CNN                       |
| DenseUNet201 | 2D, 3D               | Segmentation   | CNN                       |
| UNETR        | 2D, 3D               | Segmentation   | Transformer               |
| SwinUNETR    | 2D, 3D               | Segmentation   | Transformer               |
| TransUNet    | 2D, 3D               | Segmentation   | Transformer |
| SegFormer    | 2D, 3D               | Segmentation   | Transformer |

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

