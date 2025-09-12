
<img src="https://i.imgur.com/GvFKwDG.jpeg" width="500"/>

# MedicAI

The [`medicai`](https://github.com/innat/medic-ai) is a [Keras](https://keras.io/keras_3/) based library designed for medical image analysis using machine learning techniques. Its core strengths include:

- **Backend Agnostic**: Compatible with `tensorflow`, `torch`, and `jax`.
- **User-Friendly API**: High-level interface for transformations and model creation.
- **Scalable Execution**: Supports training and inference on **single/multi-GPU** and **TPU-VM** setups.
- **Essential Components**: Includes standard metrics and losses, such as Dice.
- **Optimized 3D Inference**: Offers an efficient sliding-window method and callback for volumetric data


## 🛠 Installation

PyPI version:

```bash
pip install medicai
```

Installing from source GitHub:

```bash
pip install git+https://github.com/innat/medic-ai.git
```

## 🍁 Available Features

The `medicai` library provides a range of features for medical image processing, model training, and inference. Below is an overview of its key functionalities.

**Image Transformations**

`medicai` includes various transformation utilities for preprocessing medical images:

- Basic Transformations:
    - `Resize`: Adjusts the image dimensions.
    - `ScaleIntensityRange`: Normalizes intensity values within a specified range.
    - `CropForeground`: Crops the image to focus on the region of interest.
    - `Spacing`: Resamples the image to a target voxel spacing.
    - `Orientation`: Standardizes image orientation.
    - `NormalizeIntensity`: Normalize the intensity of tensors based on global or channel-wise statistics.
    - `SignalFillEmpty`: Fills `nan`, positive infinity, and negative infinity values in specified tensors with a
    given replacement.
- Augmentations for Robustness:
    - `RandCropByPosNegLabel`: Randomly crops based on positive and negative label ratios.
    - `RandRotate90`: Randomly rotates images by 90 degrees.
    - `RandShiftIntensity`: Randomly shifts intensity values.
    - `RandFlip`: Randomly flips images along specified axes.
    - `RandomSpatialCrop`: Randomly crops a region of interest (ROI).
- Pipeline Composition:
     - `Compose`: Chains multiple transformations into a single pipeline.

**Models**

Currently, `medicai` focuses on 3D models for classification and segmentation:

- `SwinTransformer` – 3D classification task.
- `SwinUNETR` – 3D segmentation task.

**Inference**

- `SlidingWindowInference` – Processes large 3D images in smaller overlapping windows, improving performance and memory efficiency.

## 💡 Guides

**Segmentation**: Available guides for 3D segmentation task.

| Task | GitHub | Kaggle |
|----------|----------|----------|
| Covid-19  | <a target="_blank" href="https://github.com/innat/medic-ai/blob/main/notebooks/covid19.ct.segment.ipynb"><img src="https://img.shields.io/badge/GitHub-View%20source-lightgrey" /></a>     | <a target="_blank" href="https://www.kaggle.com/code/ipythonx/medicai-covid-19-3d-image-segmentation/notebook"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>     |
| BTCV  | <a target="_blank" href="https://github.com/innat/medic-ai/blob/main/notebooks/btcv.segment.ipynb"><img src="https://img.shields.io/badge/GitHub-View%20source-lightgrey" /></a>    | <a target="_blank" href="https://www.kaggle.com/code/ipythonx/medicai-3d-btcv-segmentation-in-keras/"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>    | 
| BraTS  | <a target="_blank" href="https://github.com/innat/medic-ai/blob/main/notebooks/brats.multi-gpu.segment.ipynb"><img src="https://img.shields.io/badge/GitHub-View%20source-lightgrey" /></a>     | <a target="_blank" href="https://www.kaggle.com/code/ipythonx/3d-brats-segmentation-in-keras-multi-gpu/"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>    |
| Spleen | <a target="_blank" href="https://github.com/innat/medic-ai/blob/main/notebooks/spleen.segment.ipynb"><img src="https://img.shields.io/badge/GitHub-View%20source-lightgrey" /></a>     | <a target="_blank" href="https://www.kaggle.com/code/ipythonx/medicai-spleen-3d-segmentation-in-keras"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>     | 

**Classification**: Available guides for 3D classification task.

| Task (Classification) | GitHub | Kaggle |
|----------|----------|----------|
| Covid-19   | <a target="_blank" href="https://github.com/innat/medic-ai/blob/main/notebooks/covid19.ct.classification.ipynb"><img src="https://img.shields.io/badge/GitHub-View%20source-lightgrey" /></a>       | <a target="_blank" href="https://www.kaggle.com/code/ipythonx/medicai-3d-image-classification"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>     |

## 🤝 Contributing

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

## 🙏 Acknowledgements

This project is greatly inspired by [MONAI](https://monai.io/).

## 📝 Citation

If you use `medicai` in your research or educational purposes, please cite it using the metadata from our [`CITATION.cff`](https://github.com/innat/medic-ai/blob/main/CITATION.cff) file.
