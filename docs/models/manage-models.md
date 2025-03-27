
Currently only two models are implemented for 3D classification and segmentation task.


## 3D

**Classification**

```python
from medicai.models import SwinTransformer

model = SwinTransformer(
    input_shape=(96, 96, 96, 1), 
    num_classes=4, 
    classifier_activation=None, 
)
```

**Segmentation**

```python
from medicai.models import SwinUNETR

model = SwinUNETR(
    input_shape=(96, 96, 96, 1),
    num_classes=4,
    classifier_activation=None,
    feature_size=48,
    res_block=True,
    norm_name="instance"
)
```