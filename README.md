# Medic-AI

**Medic-AI** serves as a python-based library for medical image analysis with AI techniques.

# Installation

```bash
git clone https://github.com/innat/medic
cd medic
pip install -e . 
```

# Usages

Using **Python API**,

```python
from medicai.dataloader import APTOSDataloader
from medicai.nets import DuelAttentionNet
from medicai.utils import Configurator

master_cfg = Configurator('src/medicai/cfg/aptos.yml')
cls_cfg = master_cfg.update_cls_cfg(
    model_name='efficientnetb0',
    input_size=224,
    num_classes=5,
)
cls_cfg.dataset.path = "/mnt/c/projects/dataset"

dataloader = APTOSDataloader(cls_cfg).generator()
model = DuelAttentionNet(cls_cfg)
model.compile(...)
hist = model.fit(dataloader)

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