

<img src="assets/logo.jpg" width="500"/>


[![Palestine](https://img.shields.io/badge/Free-Palestine-white?labelColor=green)](https://twitter.com/search?q=%23FreePalestine&src=typed_query) 

![Static Badge](https://img.shields.io/badge/keras-3.9.0-darkred?style=flat) ![Static Badge](https://img.shields.io/badge/tensorflow-2.19.0-orange?style=flat) ![Static Badge](https://img.shields.io/badge/torch-2.6.0-red?style=flat) ![Static Badge](https://img.shields.io/badge/jax-0.4.23-%233399ff)

**Medic-AI** is a [Keras](https://keras.io/keras_3/) based library designed for medical image analysis using machine learning techniques. Its core strengths include:

- **Backend Agnostic:** Compatible with `tensorflow`, `torch`, and `jax`.
- **User-Friendly API:** High-level interface for transformations and model creation.
- **Scalable Execution:** Supports training and inference on **single/multi-GPU** and **TPU-VM** setups.
- **Essential Components:** Includes standard metrics and losses, such as Dice.
- **Optimized 3D Inference:** Offers an efficient sliding-window method and callback for volumetric data


# 📋 Table of Contents
1. [Installation](#-installation)
2. [Guides](#-guides)
3. [Documentation](#-documentation)
4. [Acknowledgements](#-acknowledgements)
5. [Citation](#-citation)


# 🛠 Installation

PyPI version:

```bash
!pip install medicai
```

Installing from source GitHub:

```bash
!pip install git+https://github.com/innat/medic-ai.git
```

# 💡 Guides

**Segmentation**: Available guides for 3D segmentation task.

| Task | GitHub | Kaggle | View |
|----------|----------|----------|----------|
| Covid-19  | <a target="_blank" href="notebooks/covid19.ct.segment.ipynb"><img src="https://img.shields.io/badge/GitHub-View%20source-lightgrey" /></a>     | <a target="_blank" href="https://www.kaggle.com/code/ipythonx/medicai-covid-19-3d-image-segmentation/notebook"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>     | <img src="assets/covid.gif" width="200"/>    |
| BTCV  | <a target="_blank" href="notebooks/btcv.segment.ipynb"><img src="https://img.shields.io/badge/GitHub-View%20source-lightgrey" /></a>    | <a target="_blank" href="https://www.kaggle.com/code/ipythonx/medicai-3d-btcv-segmentation-in-keras/"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>    | n/a     |
| BraTS  | coming soon     | coming soon     | n/a     |
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
