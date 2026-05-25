# Xception

**Xception** is a deep convolutional neural network architecture that replaces standard convolution layers with depthwise separable convolutions to improve efficiency and performance. It decouples spatial and channel-wise feature learning, allowing the model to capture more expressive representations with fewer parameters. Xception builds upon the Inception idea by fully separating cross-channel and spatial correlations, making it highly effective for image classification and recognition tasks. In ```medic-ai```, Xception supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` automatically construct 2D Xception models, while ```(depth, height, width, channel)``` enables volumetric 3D Xception models for medical and scientific imaging applications.

## XceptionBackbone

```{eval-rst}
.. autoclass:: medicai.models.XceptionBackbone
```

## Xception

```{eval-rst}
.. autoclass:: medicai.models.Xception
```
