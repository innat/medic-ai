# TransUNet

**TransUNet** is a hybrid segmentation architecture that combines CNN-based feature extraction with Transformer-based global context modeling in a U-shaped design. It enhances U-Net by using self-attention to capture long-range dependencies while preserving local spatial details through skip connections. The model is effective for medical image segmentation, especially where both fine structures and global context are important. In ```medic-ai```, TransUNet supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` construct 2D models, while ```(depth, height, width, channel)``` enables 3D volumetric models for medical and scientific imaging applications.

```{eval-rst}
.. note::

   There are two published **TransUNet** variants for ``2D`` and ``3D``
   tasks, and they differ in decoder design.

   In the 3D **TransUNet** paper, the authors note that the model can also be
   applied to ``2D`` tasks by replacing ``3D`` operations with their ``2D``
   counterparts.

   In this codebase, the implemented architecture follows the
   `3D TransUNet <https://arxiv.org/abs/2310.07781>`_ formulation. If a
   ``2D`` input shape is provided, the built model is the ``2D`` adaptation
   of that 3D design.
```

## TransUNet

```{eval-rst}
.. autoclass:: medicai.models.TransUNet
```
