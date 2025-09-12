
The `medicai` provides **Swin Transformer** and **SwinUNETR** models for 3D classification and segmentation respectively. These models are translated from official release to keras, and able to run on multiple backend, i.e., `tensorflow`, `torch`, and, `jax` backends.


## 3D Models

**Classification**

```python
import tensorflow as tf
from medicai.models import SwinTransformer

num_classes = 4
input_shape = (96, 96, 96, 1)
model = SwinTransformer(
    input_shape=input_shape, 
    num_classes=num_classes, 
    classifier_activation=None
)

dummy_input = tf.random.normal((1, 96, 96, 96, 1))
output = model(dummy_input)
output.shape
TensorShape([1, 4])
```

**Segmentation**

```python
import tensorflow as tf
from medicai.models import SwinUNETR

num_classes = 4
input_shape = (96, 96, 96, 1)
model = SwinUNETR(
    input_shape=input_shape, 
    num_classes=num_classes,
    classifier_activation=None
)

dummy_input = tf.random.normal((1, 96, 96, 96, 1))
output = model(dummy_input)
output.shape
TensorShape([1, 96, 96, 96, 4])
```