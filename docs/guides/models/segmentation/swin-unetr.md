# SwinUNETR

**Swin UNETR** is a hybrid segmentation architecture that integrates the Swin Transformer as an encoder with a UNet-style decoder for volumetric medical image segmentation. It leverages shifted window self-attention to efficiently model long-range dependencies while preserving hierarchical feature representations. The UNet decoder progressively reconstructs spatial resolution using skip connections from the transformer encoder. In ```medic-ai```, Swin UNETR supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` automatically construct 2D Swin UNETR models, while ```(depth, height, width, channel)``` enables volumetric 3D Swin UNETR models for medical and scientific imaging applications.

## SwinUNETR

```{eval-rst}
.. autoclass:: medicai.models.SwinUNETR
```
