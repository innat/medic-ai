# UNETR++

**UNETR++** is an improved version of UNETR that enhances feature fusion between transformer encoder and decoder using more efficient skip connection and hierarchical refinement strategies. It improves multi-scale representation learning by strengthening interaction between global transformer features and local convolutional decoding. This leads to better boundary accuracy and robustness in complex segmentation tasks. In ```medic-ai```, UNETR++ supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` construct 2D models, while ```(depth, height, width, channel)``` enables 3D volumetric models for medical and scientific imaging applications.

## UNETRPlusPlusEncoder

```{eval-rst}
.. autoclass:: medicai.models.UNETRPlusPlusEncoder
```

## UNETRPlusPlus

```{eval-rst}
.. autoclass:: medicai.models.UNETRPlusPlus
```
