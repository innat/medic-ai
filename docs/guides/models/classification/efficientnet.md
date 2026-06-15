# EfficientNet

**EfficientNet** is a convolutional neural network architecture that introduces a compound scaling strategy to balance network depth, width, and input resolution efficiently. Instead of arbitrarily scaling individual dimensions, EfficientNet uniformly scales the entire model to achieve better accuracy and computational efficiency across different model sizes. The architecture uses mobile inverted bottleneck convolution (MBConv) blocks and squeeze-and-excitation mechanisms to improve feature representation while maintaining a lightweight design. In ```medic-ai```, EfficientNet supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` automatically construct 2D EfficientNet models, while ```(depth, height, width, channel)``` enables volumetric 3D EfficientNet models for medical and scientific imaging applications.

## EfficientNetBackbone

```{eval-rst}
.. autoclass:: medicai.models.EfficientNetBackbone
```

## EfficientNetBackboneV2

```{eval-rst}
.. autoclass:: medicai.models.EfficientNetBackboneV2
```

## EfficientNetB0

```{eval-rst}
.. autoclass:: medicai.models.EfficientNetB0
```

## EfficientNetB1

```{eval-rst}
.. autoclass:: medicai.models.EfficientNetB1
```

## EfficientNetB2

```{eval-rst}
.. autoclass:: medicai.models.EfficientNetB2
```

## EfficientNetB3

```{eval-rst}
.. autoclass:: medicai.models.EfficientNetB3
```

## EfficientNetB4

```{eval-rst}
.. autoclass:: medicai.models.EfficientNetB4
```

## EfficientNetB5

```{eval-rst}
.. autoclass:: medicai.models.EfficientNetB5
```

## EfficientNetB6

```{eval-rst}
.. autoclass:: medicai.models.EfficientNetB6
```

## EfficientNetB7

```{eval-rst}
.. autoclass:: medicai.models.EfficientNetB7
```

## EfficientNetB8

```{eval-rst}
.. autoclass:: medicai.models.EfficientNetB8
```

## EfficientNetL2

```{eval-rst}
.. autoclass:: medicai.models.EfficientNetL2
```

## EfficientNetV2B0

```{eval-rst}
.. autoclass:: medicai.models.EfficientNetV2B0
```

## EfficientNetV2B1

```{eval-rst}
.. autoclass:: medicai.models.EfficientNetV2B1
```

## EfficientNetV2B2

```{eval-rst}
.. autoclass:: medicai.models.EfficientNetV2B2
```

## EfficientNetV2B3

```{eval-rst}
.. autoclass:: medicai.models.EfficientNetV2B3
```

## EfficientNetV2S

```{eval-rst}
.. autoclass:: medicai.models.EfficientNetV2S
```

## EfficientNetV2M

```{eval-rst}
.. autoclass:: medicai.models.EfficientNetV2M
```

## EfficientNetV2L

```{eval-rst}
.. autoclass:: medicai.models.EfficientNetV2L
```
