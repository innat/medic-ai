# EyeNet

EyeNet serves as a sophisticated Python-based library, fundamentally conceived to cater to the domain of ophthalmic diseases. Primarily, this toolset aims to provide a comprehensive platform to unravel, investigate, and interpret complexities related to various eye diseases.

# Installation

```python
git clone https://github.com/innat/eye-net
cd eye-net
pip install -e . 
```

# Usages

Using **Python API**,

```python
from eyenet.dataloader import APTOSDataloader
from eyenet.nets import DuelAttentionNet
from eyenet.utils import MasterConfigurator

master_cfg = MasterConfigurator('eyenet/cfg/aptos.yml')
cls_cfg = master_cfg.get_cls_cfg(
    model_name='efficientnetb0',
    input_size=224,
    num_classes=5,
    metrics='cohen_kappa',
    losses='cohen_kappa',
)
dataloader = APTOSDataloader(cls_cfg)
model = DuelAttentionNet(cls_cfg)
hist = model.fit(dataloader.load())

>>> hist.history
{'loss': [-0.09087201207876205], 'cohen_kappa': [0.0476190447807312]}
```

Using **CLI**,

```python
eyenet train --config "eyenet/cfg/aptos.yml" 
eyenet inference --image-path "dataset/aptos/00a8624548a9.png"
```


# Docker [cpu]

```python
docker build -f docker/Dockerfile-cpu -t eyenetapp:cpu .
docker run -it --rm eyenetapp:cpu
```

# Test 

```python
python -m pytest test/unit/config/test_config.py
python -m pytest test/unit/net/test_segmentation_predict.py

python -m pytest test/integration/training/test_cls_training.py
python -m pytest test/integration/training/test_seg_training.py
```