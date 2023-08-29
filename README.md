# Medic-AI

**Medic-AI** serves as a python-based library for medical image analysis with AI techniques.

# Installation

```bash
git clone https://github.com/innat/medic-ai
cd medic-ai
pip install -e . 
```

# Usages

Using **Python API**,

```python
from tensorflow import keras
from medicai.datasets import APTOSDataloader
from medicai.nets import UNet2D
from medicai.losses import WeightedKappaLoss
from medical.metrics import CohenKappa

# build dataloader
db = APTOSDataloader(
    Path('/mnt/c/Users/innat/Desktop/projects/dataset/aptos'), 
    subfolder='train_images', 
    meta_file='df.csv', 
    meta_columns=['id_code', 'diagnosis'], 
    num_classes=5,
    label_mode='categorical',
    batch_size=8
)

# build the model and compile
model = UNet2D(
    backbone='efficientnetb0',
    input_size=224,
    num_classes=5,
    class_activation='sigmoid'
)

model.compile(
    loss=WeightedKappaLoss(),
    metrics=CohenKappa(num_classes=5),
    optimizer='adam'
)
hist = model.fit(data, epochs=cls_cfg.trainer.epochs)

>>> hist.history
{'loss': [-0.09087201207876205], 'cohen_kappa': [0.0476190447807312]}
```

Using **CLI**,

The medic can be run on command line interface.

```python
medic train --config "medicai/cfg/aptos.yml" 
medic inference --image-path "dataset/aptos/00a8624548a9.png"
```

# Docker 

MEDIC can be run with container. First build the docker image.

```bash
docker build -f docker/Dockerfile -t medicai:cpu .
```

Next, run the container.

```bash
docker run 
-it \
--rm \
-v {absoluate_path}/dataset:/app/dataset \
-v {absoluate_path}/results:/app/results \
medicai:cpu
```

It will give an interactive python session which enable running the Python SDK.