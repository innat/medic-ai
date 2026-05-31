# Losses

Loss functions exported by `medicai.losses`.

## BaseDiceLoss

```{eval-rst}
.. autoclass:: medicai.losses.BaseDiceLoss
   :members: __init__, compute_loss
```

## Dice

```{eval-rst}
.. autoclass:: medicai.losses.BinaryDiceLoss
   :members: __init__, compute_loss

.. autoclass:: medicai.losses.CategoricalDiceLoss
   :members: __init__, compute_loss

.. autoclass:: medicai.losses.SparseDiceLoss
   :members: __init__, compute_loss
```

## Dice Cross-Entropy

```{eval-rst}
.. autoclass:: medicai.losses.BinaryDiceCELoss
   :members: __init__, compute_loss

.. autoclass:: medicai.losses.CategoricalDiceCELoss
   :members: __init__, compute_loss

.. autoclass:: medicai.losses.SparseDiceCELoss
   :members: __init__, compute_loss
```

## Generalized Dice

```{eval-rst}
.. autoclass:: medicai.losses.BinaryGeneralizedDiceLoss
   :members: __init__, compute_loss

.. autoclass:: medicai.losses.CategoricalGeneralizedDiceLoss
   :members: __init__, compute_loss

.. autoclass:: medicai.losses.SparseGeneralizedDiceLoss
   :members: __init__, compute_loss
```

## IoU

```{eval-rst}
.. autoclass:: medicai.losses.BinaryIoULoss
   :members: __init__, compute_loss

.. autoclass:: medicai.losses.CategoricalIoULoss
   :members: __init__, compute_loss

.. autoclass:: medicai.losses.SparseIoULoss
   :members: __init__, compute_loss
```

## Tversky

```{eval-rst}
.. autoclass:: medicai.losses.BinaryTverskyLoss
   :members: __init__, compute_loss

.. autoclass:: medicai.losses.CategoricalTverskyLoss
   :members: __init__, compute_loss

.. autoclass:: medicai.losses.SparseTverskyLoss
   :members: __init__, compute_loss
```

## Centerline Dice

```{eval-rst}
.. autoclass:: medicai.losses.BinaryCenterlineDiceLoss
   :members: __init__, compute_loss

.. autoclass:: medicai.losses.CategoricalCenterlineDiceLoss
   :members: __init__, compute_loss

.. autoclass:: medicai.losses.SparseCenterlineDiceLoss
   :members: __init__, compute_loss
```
