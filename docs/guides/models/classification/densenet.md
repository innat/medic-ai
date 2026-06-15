# DenseNet

**DenseNet** is a convolutional neural network architecture that introduces dense connections between layers to improve feature reuse and gradient propagation. Each layer receives feature maps from all preceding layers, allowing the network to preserve low-level and high-level representations throughout the model. In ```medic-ai```, DenseNet supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` automatically construct 2D DenseNet models, while ```(depth, height, width, channel)``` enables volumetric 3D DenseNet models for medical and scientific imaging applications.


## DenseNetBackbone

```{eval-rst}
.. autoclass:: medicai.models.DenseNetBackbone
```

## DenseNet121

```{eval-rst}
.. autoclass:: medicai.models.DenseNet121
```


## DenseNet169

```{eval-rst}
.. autoclass:: medicai.models.DenseNet169
```


## DenseNet201

```{eval-rst}
.. autoclass:: medicai.models.DenseNet201
```

## DenseNet264

```{eval-rst}
.. autoclass:: medicai.models.DenseNet264
```