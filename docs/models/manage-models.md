
Currently only two models are implemented for 3D classification and segmentation task. The workflow can be run with `tensorflow`, and `torch` backend.


## 3D

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