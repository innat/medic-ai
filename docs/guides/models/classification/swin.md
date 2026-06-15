# Swin

**Swin Transformer** is a hierarchical vision transformer architecture that introduces shifted window-based self-attention for efficient and scalable feature learning. Instead of applying global self-attention across the entire image, Swin Transformer computes attention within local windows while periodically shifting window partitions to enable cross-window information exchange. This design significantly reduces computational complexity while preserving strong representation capability for classification, detection, and segmentation tasks. In ```medic-ai```, Swin Transformer supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` automatically construct 2D Swin Transformer models, while ```(depth, height, width, channel)``` enables volumetric 3D Swin Transformer models for medical and scientific imaging applications.

```{eval-rst}
.. note::

   **What distinguishes Swin V1 from Swin V2?**

   Both variants use the same hierarchical Swin Transformer design:

   - patch embedding to convert the input into non-overlapping patches
   - dropout after patch embedding
   - multiple Swin stages built from shifted-window attention and MLP blocks
   - optional patch merging between stages
   - support for both ``2D`` and ``3D`` inputs
   - optional ``stage_wise_conv=True`` blocks following the SwinUNETR-V2 style

   **Swin V1**

   - Uses the original ``SwinBasicLayer`` blocks.
   - Uses the original patch merging formulation.

   **Swin V2**

   - Uses ``SwinBasicLayerV2`` with scaled cosine attention.
   - Uses log-spaced continuous relative position bias.
   - Uses the V2 block design with improved numerical stability.
   - Uses ``SwinPatchMergingV2`` for a more stable patch merging step.

   In short, Swin V2 keeps the overall Swin architecture the same, but updates
   the attention, normalization, and patch merging details to improve training
   stability, especially at larger scales.
```

## SwinBackbone

```{eval-rst}
.. autoclass:: medicai.models.SwinBackbone
```

## SwinBackboneV2

```{eval-rst}
.. autoclass:: medicai.models.SwinBackboneV2
```

## SwinTiny

```{eval-rst}
.. autoclass:: medicai.models.SwinTiny
```

## SwinSmall

```{eval-rst}
.. autoclass:: medicai.models.SwinSmall
```

## SwinBase

```{eval-rst}
.. autoclass:: medicai.models.SwinBase
```

## SwinTinyV2

```{eval-rst}
.. autoclass:: medicai.models.SwinTinyV2
```

## SwinSmallV2

```{eval-rst}
.. autoclass:: medicai.models.SwinSmallV2
```

## SwinBaseV2

```{eval-rst}
.. autoclass:: medicai.models.SwinBaseV2
```
