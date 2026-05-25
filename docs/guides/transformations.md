# Transformations

Code-driven API reference for `medicai.transforms`.

These transformations are designed for medical imaging workflows and are
implemented with TensorFlow-compatible operations so they can be used in
`tf.data` pipelines as well as broader Keras 3 multi-backend workflows.

For an end-to-end example, see the Kaggle notebook:
[3D Transformation](https://www.kaggle.com/code/ipythonx/medicai-3d-medical-image-transformation).

## Pipeline

```{eval-rst}
.. autoclass:: medicai.transforms.Compose
   :members: __init__
```

## Spatial Transforms

```{eval-rst}
.. autoclass:: medicai.transforms.Resize
   :members: __init__

.. autoclass:: medicai.transforms.Spacing
   :members: __init__

.. autoclass:: medicai.transforms.Orientation
   :members: __init__

.. autoclass:: medicai.transforms.CropForeground
   :members: __init__

.. autoclass:: medicai.transforms.RandCropByPosNegLabel
   :members: __init__

.. autoclass:: medicai.transforms.RandSpatialCrop
   :members: __init__
```

## Intensity Transforms

```{eval-rst}
.. autoclass:: medicai.transforms.NormalizeIntensity
   :members: __init__

.. autoclass:: medicai.transforms.ScaleIntensityRange
   :members: __init__

.. autoclass:: medicai.transforms.RandShiftIntensity
   :members: __init__
```

## Random Augmentations

```{eval-rst}
.. autoclass:: medicai.transforms.RandFlip
   :members: __init__

.. autoclass:: medicai.transforms.RandRotate90
   :members: __init__

.. autoclass:: medicai.transforms.RandRotate
   :members: __init__

.. autoclass:: medicai.transforms.RandCutOut
   :members: __init__
```

## Data Container

```{eval-rst}
.. autoclass:: medicai.transforms.TensorBundle
   :members: __init__, get_data, get_meta, set_data, set_meta
```
