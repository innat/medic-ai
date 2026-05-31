# Transformations

`medicai.transforms` are designed for medical imaging workflows and are implemented with TensorFlow-compatible operations so they can be used in `tf.data` pipelines.

Most transforms accept either a sample dictionary or a `TensorBundle` and
return a `TensorBundle`. The transformed tensors stay under the same keys, so
if we pass inputs such as `image` and `label`, we retrieve the outputs using
those same keys after each transform or composed pipeline.


## Data Container

```{eval-rst}
.. autoclass:: medicai.transforms.TensorBundle
```

## Spatial Transforms

```{eval-rst}
.. autoclass:: medicai.transforms.Resize

.. autoclass:: medicai.transforms.Spacing

.. autoclass:: medicai.transforms.Orientation

.. autoclass:: medicai.transforms.CropForeground

.. autoclass:: medicai.transforms.RandCropByPosNegLabel

.. autoclass:: medicai.transforms.RandSpatialCrop
```

## Intensity Transforms

```{eval-rst}

.. autoclass:: medicai.transforms.NormalizeIntensity

.. autoclass:: medicai.transforms.ScaleIntensityRange

.. autoclass:: medicai.transforms.RandShiftIntensity
```

## Random Augmentations

```{eval-rst}
.. autoclass:: medicai.transforms.RandFlip

.. autoclass:: medicai.transforms.RandRotate90

.. autoclass:: medicai.transforms.RandRotate

.. autoclass:: medicai.transforms.RandCutOut
```

## Pipeline

```{eval-rst}
.. autoclass:: medicai.transforms.Compose
```
