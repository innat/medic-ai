# Utility

Utility classes and functions exported by `medicai.utils`.

## Sliding Window Inference

```{eval-rst}
.. autoclass:: medicai.utils.SlidingWindowInference
.. autofunction:: medicai.utils.sliding_window_inference
.. autofunction:: medicai.utils.extract_patches
.. autofunction:: medicai.utils.merge_patches
.. autofunction:: medicai.utils.predict_patches
```

## GradCAM

```{eval-rst}
.. autoclass:: medicai.utils.GradCAM
   :members: compute_heatmap
```

## Image Utilities

```{eval-rst}
.. autofunction:: medicai.utils.resize_volumes
```

## Loss Utilities

```{eval-rst}
.. autofunction:: medicai.utils.soft_skeletonize
```

## Model Utilities

```{eval-rst}
.. autofunction:: medicai.utils.get_act_layer

.. autofunction:: medicai.utils.get_conv_layer

.. autofunction:: medicai.utils.get_dropout_layer

.. autofunction:: medicai.utils.get_norm_layer

.. autofunction:: medicai.utils.get_pooling_layer

.. autofunction:: medicai.utils.get_reshaping_layer
```
