# DeepLabV3Plus

**DeepLabV3+** is a semantic segmentation architecture designed to produce accurate pixel-level predictions by combining atrous spatial pyramid pooling (ASPP) with an encoder–decoder structure. It captures multi-scale contextual information using dilated (atrous) convolutions while preserving spatial resolution efficiently. The decoder refines segmentation boundaries by progressively recovering spatial details, making it highly effective for complex scene understanding and medical image segmentation. In ```medic-ai```, DeepLabV3+ supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` automatically construct 2D DeepLabV3+ models, while ```(depth, height, width, channel)``` enables volumetric 3D DeepLabV3+ models for medical and scientific imaging applications.

## DeepLabV3Plus

```{eval-rst}
.. autoclass:: medicai.models.DeepLabV3Plus
```
