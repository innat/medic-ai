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
from medicai.metrics import DiceMetric
from medicai.utils import SlidingWindowInference

# build dataloader
dataloader = ...
num_classes=4

# build the model, compile
model = SwinUNETR(
    input_shape=(96, 96, 96, 1),
    num_classes=4,
)
model.compile(
    loss=DiceCELoss(to_onehot_y=True, softmax=True),
    metrics=[
        DiceCoefficient3D(
            num_classes,
            include_background=True,
            reduction="mean",
            ignore_empty=True,
            smooth=1e-6,
            name='dice_score'
        )
    ],
    optimizer='adamw'
)

# train the model
hist = model.fit(dataloader, epochs=10)

# evaluation
val_ds = ...
input, label = next(iter(val_dataloader))
swi = SlidingWindowInference(
    model
    num_classes=num_classes, 
    roi_size=(96, 96, 96), 
    sw_batch_size=4, 
    overlap=0.8
)
pred = swi(input)
dice_metric = DiceCoefficient3D(
    num_classes=num_classes,
    include_background=True,
    reduction="mean",
    ignore_empty=True,
    smooth=1e-6,
    name='dice_score'
)
dice_metric.update_state(y, output)
dice_score = dice_metric.result()
print(f"Dice Score: {dice_score.numpy()}")
Dice Score: 0.73
```

![](src/medicai/assets/sample_predict.png)