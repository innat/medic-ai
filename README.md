# EyeNet

```python
git clone https://github.com/innat/eye-net
cd eye-net
pip install -e . 
```

Using python API,

```python

from eyenet import get_configured
from eyenet import get_dataloader
from eyenet import get_model

cfg = get_configured('eyenet/cfg/default.yml')
db = get_dataloader(cfg)
model = get_model(cfg)
hist = model.fit(db)

>>> hist.history
{
    'loss': [0.4282873570919037], 
    'primary_loss': [-0.018353674560785294], 
    'auxilary_loss': [1.4888033866882324], 
    'primary_accuracy': [0.375], 
    'primary_primary_metrics': [-0.17293238639831543], 
    'auxilary_auxilary_metrics': [0.625]
}
```

Using CLI,

```python
eyenet train --config "eyenet/cfg/default.yml" 
eyenet inference --image-path "dataset/aptos/00a8624548a9.png"
```