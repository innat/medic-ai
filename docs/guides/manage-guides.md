
# Code Examples

The following guides provide comprehensive, end-to-end examples covering data loading, model training, and evaluation workflows. You can use various types of data loaders, including [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset), [`keras.utils.PyDataset`](https://keras.io/api/utils/python_utils/#pydataset-class), [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html), or even a custom Python generator function. These workflows are designed to be flexible and can run seamlessly on a single device or scale across multiple **GPUs** or **TPUs**, depending on your setup.

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