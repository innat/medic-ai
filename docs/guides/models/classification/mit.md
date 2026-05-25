# MiT

**Mix Vision Transformer (MiT)** is a hierarchical transformer-based architecture designed for efficient multi-scale feature extraction in computer vision tasks. It combines overlapping patch embeddings with transformer encoder blocks to preserve local spatial information while learning global contextual representations across different stages. The hierarchical design progressively reduces spatial resolution and increases feature dimensions, making MiT particularly effective for dense prediction tasks such as segmentation and detection. In ```medic-ai```, Mix Vision Transformer supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` automatically construct 2D MiT models, while ```(depth, height, width, channel)``` enables volumetric 3D MiT models for medical and scientific imaging applications.

## MiTBackbone

```{eval-rst}
.. autoclass:: medicai.models.MiTBackbone
```

## MixViTB0

```{eval-rst}
.. autoclass:: medicai.models.MixViTB0
```

## MixViTB1

```{eval-rst}
.. autoclass:: medicai.models.MixViTB1
```

## MixViTB2

```{eval-rst}
.. autoclass:: medicai.models.MixViTB2
```

## MixViTB3

```{eval-rst}
.. autoclass:: medicai.models.MixViTB3
```

## MixViTB4

```{eval-rst}
.. autoclass:: medicai.models.MixViTB4
```

## MixViTB5

```{eval-rst}
.. autoclass:: medicai.models.MixViTB5
```
