# MEDIC

MEDIC serves as a python-based library for medical image analysis.

# Installation

```bash
git clone https://github.com/innat/medic
cd medic
pip install -e . 
```

# Usages

Using **Python API**,

```python
from medic.dataloader import APTOSDataloader
from medic.nets import DuelAttentionNet
from medic.utils import MasterConfigurator

master_cfg = MasterConfigurator('medic/cfg/aptos.yml')
cls_cfg = master_cfg.get_cls_cfg(
    model_name='efficientnetb0',
    input_size=224,
    num_classes=5,
    metrics='cohen_kappa',
    losses='cohen_kappa',
)
dataloader = APTOSDataloader(cls_cfg).load()
model = DuelAttentionNet(cls_cfg)
hist = model.fit(dataloader)

>>> hist.history
{'loss': [-0.09087201207876205], 'cohen_kappa': [0.0476190447807312]}
```

Using **CLI**,

The medic can be run on command line interface.

```python
medic train --config "medic/cfg/aptos.yml" 
medic inference --image-path "dataset/aptos/00a8624548a9.png"
```

# Docker 

MEDIC can be run with container. First build the docker image.

```bash
docker build -f docker/Dockerfile -t medicapp:cpu .
```

Next, run the container.

```bash
docker run 
-it \
--rm \
-v {absoluate_path}/dataset:/app/dataset \
-v {absoluate_path}/results:/app/results \
eyenetapp:cpu
```

It will give an interactive python session which enable running the Python SDK.