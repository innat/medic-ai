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
from medicai.nets import DuelAttentionNet
from medicai.utils import Configurator
from medicai.losses import WeightedKappaLoss
from medical.metrics import CohenKappa

# load and update config
loaded_cfg = Configurator('src/medicai/cfg/aptos.yml')
cls_cfg = loaded_cfg.update_cls_cfg(
    model_name='efficientnetb0',
    input_size=224,
    num_classes=5,
)
cls_cfg.dataset.path = "/mnt/c/projects/dataset"
cls_cfg.trainer.learning_rate = 0.003
cls_cfg.trainer.epochs = 10

# build data-loader
data = APTOSDataloader(cls_cfg)
data = data.preprocess() # ((h,w,3), (num_class,))
data = data.prepare_batches() # ((bs, h,w,3), (bs, num_class,))

# build the model and compile
model = DuelAttentionNet(cls_cfg)
model.compile(
    loss=WeightedKappaLoss(),
    metrics=CohenKappa(
        num_classes=cls_cfg.dataset.num_classes, 
        name="cohen_kappa"
    ),
    optimizer=keras.optimizers.Adam(
        learning_rate=cls_cfg.trainer.learning_rate
    )
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