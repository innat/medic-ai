# SegFormer

**SegFormer** is a transformer-based semantic segmentation architecture that combines a hierarchical Transformer encoder with a lightweight MLP decoder for efficient dense prediction. It avoids complex decoding structures and instead relies on multi-scale feature representations learned through progressive patch merging. This design makes SegFormer both computationally efficient and highly accurate across diverse segmentation tasks. In ```medic-ai```, SegFormer supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` automatically construct the 2D SegFormer variant described in [arXiv:2105.15203](https://arxiv.org/abs/2105.15203), while ```(depth, height, width, channel)``` enables the volumetric 3D SegFormer variant described in [arXiv:2404.10156](https://arxiv.org/abs/2404.10156) for medical and scientific imaging applications.

```{eval-rst}
.. note::

   **SegFormer 2D**

   - Designed for general-purpose ``2D`` semantic segmentation.
   - Uses Mix Vision Transformer (**MiT**) as the encoder backbone.
   - Supports six **MiT** variants: ``mit_b0`` through ``mit_b5``.
   - When the input is ``2D`` (for example, ``(H, W, C)``), this
     implementation automatically builds the ``2D`` SegFormer model.

   **SegFormer 3D**

   - Extends the ``2D`` SegFormer to handle volumetric (``3D``) data by
     replacing ``2D`` operations with ``3D`` counterparts.
   - Uses modified spatial reduction ratios:

     - ``2D``: ``[8, 4, 2, 1]``
     - ``3D``: ``[4, 2, 1, 1]``

   - Officially demonstrated using only the ``mit_b0`` backbone.
   - For other **MiT** variants (``mit_b1`` through ``mit_b5``), this
     implementation uses the same reduction strategy for compatibility.
```

## SegFormer

```{eval-rst}
.. autoclass:: medicai.models.SegFormer
```
