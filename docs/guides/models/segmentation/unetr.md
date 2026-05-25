# UNETR

**UNETR** is a transformer-based segmentation architecture that replaces the traditional CNN encoder with a Vision Transformer for global representation learning. It extracts patch embeddings from volumetric inputs and uses a U-Net-style decoder to recover spatial resolution via skip connections. This hybrid design enables strong global context modeling while preserving localization ability for medical image segmentation. In ```medic-ai```, UNETR supports both 2D and 3D variants depending on the provided input shape. Inputs of shape ```(height, width, channel)``` construct 2D models, while ```(depth, height, width, channel)``` enables 3D volumetric models for medical and scientific imaging applications.

```{eval-rst}
.. note::

   **How UNETR selects skip layers**

   UNETR does not take skip features from every entry in
   ``encoder.pyramid_outputs``. Instead, it selects **three intermediate
   transformer features** plus the **final encoder output**.

   Let:

   - ``N`` = total number of transformer layers
   - ``d`` = number of skip features to sample
   - ``i`` = skip index, from ``1`` to ``d``

   Then the selected pyramid level is:

   .. math::

      P_i^{(\mathrm{skip})}
      =
      P_{\mathrm{round}\left(\frac{N i}{d + 1}\right) + 2}

   Intuitively, this samples the transformer at evenly spaced depths such as
   one-quarter, one-half, and three-quarters of the encoder.

   For the default UNETR case with ``N = 12`` and ``d = 3``:

   .. math::

      [P_{\mathrm{skip}}] = [P5,\; P8,\; P11]

   The final encoder output is then used as the bottleneck representation.

   This means a custom ViT encoder can still work, but it must provide:

   - encoder's ``pyramid_outputs`` dictionary containing the required sampled token
     features
   - ``patch_size`` so token sequences can be reshaped back into spatial
     feature maps
   - ``hidden_dim`` so the decoder can build its projection blocks
   - ``pooling=None`` and no class-token-only output, since the decoder needs
     the full token sequence

   In practice, changing the number of transformer layers changes which
   ``P``-keys are required by the decoder, because the skip selection is based
   on the encoder depth.
```

## UNETR

```{eval-rst}
.. autoclass:: medicai.models.UNETR
```
