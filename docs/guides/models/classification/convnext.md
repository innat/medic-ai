# ConvNeXt

**ConvNeXt** is a modern convolutional neural network architecture that incorporates transformer-inspired design choices while retaining the efficiency and simplicity of CNNs. It introduces improvements such as large kernel convolutions, inverted bottlenecks, layer normalization, and simplified network scaling to achieve strong performance across image classification and dense prediction tasks. In ```medic-ai```, ConvNeXt supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` automatically construct 2D ConvNeXt models, while ```(depth, height, width, channel)``` enables volumetric 3D ConvNeXt models for medical and scientific imaging applications.

## ConvNeXtBackbone

```{eval-rst}
.. autoclass:: medicai.models.ConvNeXtBackbone
```

## ConvNeXtBackboneV2

```{eval-rst}
.. autoclass:: medicai.models.ConvNeXtBackboneV2
```

## ConvNeXtTiny

```{eval-rst}
.. autoclass:: medicai.models.ConvNeXtTiny
```

## ConvNeXtSmall

```{eval-rst}
.. autoclass:: medicai.models.ConvNeXtSmall
```

## ConvNeXtBase

```{eval-rst}
.. autoclass:: medicai.models.ConvNeXtBase
```

## ConvNeXtLarge

```{eval-rst}
.. autoclass:: medicai.models.ConvNeXtLarge
```

## ConvNeXtXLarge

```{eval-rst}
.. autoclass:: medicai.models.ConvNeXtXLarge
```

## ConvNeXtV2Atto

```{eval-rst}
.. autoclass:: medicai.models.ConvNeXtV2Atto
```

## ConvNeXtV2Femto

```{eval-rst}
.. autoclass:: medicai.models.ConvNeXtV2Femto
```

## ConvNeXtV2Pico

```{eval-rst}
.. autoclass:: medicai.models.ConvNeXtV2Pico
```

## ConvNeXtV2Nano

```{eval-rst}
.. autoclass:: medicai.models.ConvNeXtV2Nano
```

## ConvNeXtV2Tiny

```{eval-rst}
.. autoclass:: medicai.models.ConvNeXtV2Tiny
```

## ConvNeXtV2Small

```{eval-rst}
.. autoclass:: medicai.models.ConvNeXtV2Small
```

## ConvNeXtV2Base

```{eval-rst}
.. autoclass:: medicai.models.ConvNeXtV2Base
```

## ConvNeXtV2Large

```{eval-rst}
.. autoclass:: medicai.models.ConvNeXtV2Large
```

## ConvNeXtV2Huge

```{eval-rst}
.. autoclass:: medicai.models.ConvNeXtV2Huge
```
