# ResNet

**ResNet** is a convolutional neural network architecture that introduces residual connections to enable the training of very deep neural networks effectively. These skip connections allow layers to learn residual mappings instead of complete transformations, helping reduce the vanishing gradient problem and improving optimization stability. ResNet architectures have become foundational backbones in computer vision due to their strong performance in classification, detection, and segmentation tasks. In ```medic-ai```, ResNet supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` automatically construct 2D ResNet models, while ```(depth, height, width, channel)``` enables volumetric 3D ResNet models for medical and scientific imaging applications.

## ResNetBackbone

```{eval-rst}
.. autoclass:: medicai.models.ResNetBackbone
```

## ResNet18

```{eval-rst}
.. autoclass:: medicai.models.ResNet18
```

## ResNet34

```{eval-rst}
.. autoclass:: medicai.models.ResNet34
```

## ResNet50

```{eval-rst}
.. autoclass:: medicai.models.ResNet50
```

## ResNet50v2

```{eval-rst}
.. autoclass:: medicai.models.ResNet50v2
```

## ResNet50vd

```{eval-rst}
.. autoclass:: medicai.models.ResNet50vd
```

## ResNet101

```{eval-rst}
.. autoclass:: medicai.models.ResNet101
```

## ResNet101v2

```{eval-rst}
.. autoclass:: medicai.models.ResNet101v2
```

## ResNet152

```{eval-rst}
.. autoclass:: medicai.models.ResNet152
```

## ResNet152v2

```{eval-rst}
.. autoclass:: medicai.models.ResNet152v2
```

## ResNet200vd

```{eval-rst}
.. autoclass:: medicai.models.ResNet200vd
```

**ResNeXt** is an extension of ResNet that introduces grouped convolutions and the concept of cardinality to improve representational power without significantly increasing computational cost. Instead of only scaling depth or width, ResNeXt aggregates multiple parallel transformation paths within each residual block to learn richer feature representations efficiently. This architecture achieves strong performance across various vision tasks while maintaining a simple and modular design similar to ResNet. In ```medic-ai```, ResNeXt supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` automatically construct 2D ResNeXt models, while ```(depth, height, width, channel)``` enables volumetric 3D ResNeXt models for medical and scientific imaging applications.

## ResNeXt50

```{eval-rst}
.. autoclass:: medicai.models.ResNeXt50
```

## ResNeXt101

```{eval-rst}
.. autoclass:: medicai.models.ResNeXt101
```
