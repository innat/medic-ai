# ViT

**Vision Transformer (ViT)** is a transformer-based architecture that applies the self-attention mechanism directly to image patches for visual representation learning. The model divides an image into fixed-size patches, embeds them as token sequences, and processes them through transformer encoder layers to capture global contextual relationships. Vision Transformer achieves strong performance on image classification and dense prediction tasks by leveraging large-scale pretraining and scalable transformer architectures. In ```medic-ai```, Vision Transformer supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` automatically construct 2D ViT models, while ```(depth, height, width, channel)``` enables volumetric 3D ViT models for medical and scientific imaging applications.

## ViTBackbone

```{eval-rst}
.. autoclass:: medicai.models.ViTBackbone
```

## ViTBase

```{eval-rst}
.. autoclass:: medicai.models.ViTBase
```

## ViTLarge

```{eval-rst}
.. autoclass:: medicai.models.ViTLarge
```

## ViTHuge

```{eval-rst}
.. autoclass:: medicai.models.ViTHuge
```
