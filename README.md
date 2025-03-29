
![](assets/logo.jpg)

**Medic-AI** is a [Keras](https://keras.io/keras_3/) based library designed for medical image analysis using machine learning techniques. It provides seamless compatibility with multiple backends, allowing models to run on `tensorflow`, `torch`, and `jax`.

**Note**: It is currently in its early stages and will undergo multiple iterations before reaching a stable release.

# Installation

```bash
git clone https://github.com/innat/medic-ai
cd medic-ai
pip install . -q
```

# Available Features

The `medicai` library provides a range of features for medical image processing, model training, and inference. Below is an overview of its key functionalities.

**Image Transformations**

`medicai` includes various transformation utilities for preprocessing medical images:

- Basic Transformations:
  - `Resize` – Adjusts the image dimensions.
  - `ScaleIntensityRange` – Normalizes intensity values within a specified range.
  - `CropForeground` – Crops the image to focus on the region of interest.
  - `Spacing` – Resamples the image to a target voxel spacing.
  - `Orientation` – Standardizes image orientation.
- Augmentations for Robustness:
  - `RandRotate90` – Randomly rotates images by 90 degrees.
  - `RandShiftIntensity` – Randomly shifts intensity values.
  - `RandFlip` – Randomly flips images along specified axes.
- Pipeline Composition:
  - `Compose` – Chains multiple transformations into a single pipeline.

**Models**

Currently, `medicai` focuses on 3D models for classification and segmentation:

- `SwinTransformer` – 3D classification task.
- `SwinUNETR` – 3D segmentation task.

**Inference**

- `SlidingWindowInference` – Processes large 3D images in smaller overlapping windows, improving performance and memory efficiency.

# Acknowledgements

This project is greatly inspired by [MONAI](https://monai.io/).

# Citation

If you use this videoswin implementation in your research, please cite it using the metadata from our `CITATION.cff` file.