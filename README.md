# Medic-AI

**Medic-AI** is a [Keras 3](https://keras.io/keras_3/) based library for medical image analysis with machine learning methods. It is currently in its early stages and will undergo multiple iterations before reaching a stable release.

# Installation

```bash
git clone https://github.com/innat/medic-ai
cd medic-ai
pip install -e . 
```

# Usages

```python
import keras
from medicai.nets import SwinUNETR
from medicai.losses import DiceCELoss
from medicai.metrics import DiceCoefficient
from medicai.utils import SlidingWindowInference

# build dataloader
db = ...

# build the model, compile and train
model = SwinUNETR(
    input_shape=(96, 96, 96, 1),
    num_classes=4,
    class_activation='sigmoid'
)
model.compile(
    loss=DiceCELoss(),
    metrics=DiceCoefficient(),
    optimizer='adamw'
)
hist = model.fit(data, epochs=cls_cfg.trainer.epochs)

val_ds = ...
swi = SlidingWindowInference()
pred = swi(model, val_ds, roi_size, sw_batch_size)
```
