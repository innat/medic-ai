# UPerNet

**UPerNet** (Unified Perceptual Parsing Network) is a semantic segmentation architecture that aggregates multi-scale features using a feature pyramid-based decoder. It leverages a powerful backbone to extract hierarchical representations and combines them through a pyramid pooling strategy for rich contextual understanding. The model is particularly effective for scene parsing and medical segmentation tasks due to its strong multi-scale feature fusion capability. In ```medic-ai```, UPerNet supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` construct 2D models, while ```(depth, height, width, channel)``` enables 3D volumetric models for medical and scientific imaging applications.

## UPerNet

```{eval-rst}
.. autoclass:: medicai.models.UPerNet
```
