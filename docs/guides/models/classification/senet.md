# SENet

**SE-ResNet** is an enhanced version of ResNet that integrates squeeze-and-excitation (SE) blocks to improve channel-wise feature representation. The SE mechanism adaptively recalibrates feature responses by modeling interdependencies between channels, allowing the network to focus on more informative features. By combining residual learning with channel attention, SE-ResNet achieves improved performance across image classification and dense prediction tasks. In ```medic-ai```, SE-ResNet supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` automatically construct 2D SE-ResNet models, while ```(depth, height, width, channel)``` enables volumetric 3D SE-ResNet models for medical and scientific imaging applications.

## SEResNet18

```{eval-rst}
.. autoclass:: medicai.models.SEResNet18
```

## SEResNet34

```{eval-rst}
.. autoclass:: medicai.models.SEResNet34
```

## SEResNet50

```{eval-rst}
.. autoclass:: medicai.models.SEResNet50
```

## SEResNet50v2

```{eval-rst}
.. autoclass:: medicai.models.SEResNet50v2
```

## SEResNet50vd

```{eval-rst}
.. autoclass:: medicai.models.SEResNet50vd
```

## SEResNet101

```{eval-rst}
.. autoclass:: medicai.models.SEResNet101
```

## SEResNet101v2

```{eval-rst}
.. autoclass:: medicai.models.SEResNet101v2
```

## SEResNet152

```{eval-rst}
.. autoclass:: medicai.models.SEResNet152
```

## SEResNet152v2

```{eval-rst}
.. autoclass:: medicai.models.SEResNet152v2
```

## SEResNet200vd

```{eval-rst}
.. autoclass:: medicai.models.SEResNet200vd
```

**SE-ResNeXt** is an extension of ResNeXt that combines grouped residual transformations with squeeze-and-excitation (SE) attention mechanisms for improved feature learning. The architecture enhances channel-wise feature recalibration while preserving the multi-branch aggregated transformations introduced by ResNeXt. This combination enables stronger representational capability and improved efficiency across a wide range of computer vision tasks. In ```medic-ai```, SE-ResNeXt supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` automatically construct 2D SE-ResNeXt models, while ```(depth, height, width, channel)``` enables volumetric 3D SE-ResNeXt models for medical and scientific imaging applications.

## SEResNeXt50

```{eval-rst}
.. autoclass:: medicai.models.SEResNeXt50
```

## SEResNeXt101

```{eval-rst}
.. autoclass:: medicai.models.SEResNeXt101
```
