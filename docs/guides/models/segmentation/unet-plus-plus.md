# UNet++

**U-Net++** is an enhanced version of U-Net that introduces nested and dense skip connections to improve feature fusion between encoder and decoder. This design reduces the semantic gap between feature maps at different levels, leading to more accurate segmentation results. It refines multi-scale feature learning through progressively deeper decoding paths. In ```medic-ai```, U-Net++ supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` construct 2D models, while ```(depth, height, width, channel)``` enables 3D volumetric models for medical and scientific imaging applications.

## UNetPlusPlus

```{eval-rst}
.. autoclass:: medicai.models.UNetPlusPlus
```
