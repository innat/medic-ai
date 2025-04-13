
![](assets/logo.jpg)

[![Palestine](https://img.shields.io/badge/Free-Palestine-white?labelColor=green)](https://twitter.com/search?q=%23FreePalestine&src=typed_query)


![Static Badge](https://img.shields.io/badge/keras-3.9.0-darkred?style=flat) ![Static Badge](https://img.shields.io/badge/tensorflow-2.19.0-orange?style=flat) ![Static Badge](https://img.shields.io/badge/torch-2.6.0-red?style=flat)

**Medic-AI** is a [Keras](https://keras.io/keras_3/) based library designed for medical image analysis using machine learning techniques. It provides seamless compatibility with multiple backends, allowing models to run on `tensorflow`, `torch`, and `jax`.

**Note**: It is currently in its early stages and will undergo multiple iterations before reaching a stable release.

# Installation

1. With `pip`.

```bash
pip install medicai
```

2. With source

```bash
!git clone https://github.com/innat/medic-ai
%cd medic-ai
!pip install . -q
%cd ..
```

# Guide

- 3D transformation
- [3D classification](https://www.kaggle.com/code/ipythonx/medicai-3d-image-classification)
- [3D segmentation](https://www.kaggle.com/code/ipythonx/medicai-3d-image-segmentation)


# Acknowledgements

This project is greatly inspired by [MONAI](https://monai.io/).

# Citation

If you use `medicai` in your research, please cite it using the metadata from our [`CITATION.cff`](https://github.com/innat/medic-ai/blob/main/CITATION.cff) file.
