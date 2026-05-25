# UNet

**U-Net** is a convolutional neural network architecture designed for biomedical image segmentation using an encoder–decoder structure with skip connections. The encoder captures contextual features while the decoder reconstructs spatial details for precise pixel-level prediction. Its skip connections help preserve fine-grained information, making it highly effective for medical imaging tasks. In ```medic-ai```, U-Net supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` construct 2D models, while ```(depth, height, width, channel)``` enables 3D volumetric models for medical and scientific imaging applications.

## UNet

```{eval-rst}
.. autoclass:: medicai.models.UNet
```
