# AttentionUNet

**Attention U-Net** extends the standard U-Net architecture by inserting attention gates along the decoder skip pathways. These gates help the model emphasize relevant encoder responses while suppressing noisy or irrelevant features before skip fusion. In ```medic-ai```, AttentionUNet supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` construct 2D models, while ```(depth, height, width, channel)``` enables 3D volumetric models for medical and scientific imaging applications.

## AttentionUNet

```{eval-rst}
.. autoclass:: medicai.models.AttentionUNet
```
