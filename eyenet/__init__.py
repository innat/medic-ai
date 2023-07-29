import os
import warnings

__version__ = "0.0.1"

def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from .net import get_model
from .utils import get_config
from .data import get_dataloader
from pathlib import Path
from omegaconf import OmegaConf
from tensorflow import keras

config_path = "eyenet/cfg/default.yml"
config = get_config(config_path=config_path)

project_path = Path(config.project.path) / config.dataset.name / config.model.name / "run"
(project_path / "weights").mkdir(parents=True, exist_ok=True)
(project_path / "images").mkdir(parents=True, exist_ok=True)
config.project.path = str(project_path)
# (project_path / "config.yaml").write_text(OmegaConf.to_yaml(config))
# (project_path / "config_original.yaml").write_text(OmegaConf.to_yaml(config_path))

dataloader = get_dataloader(config)
model = get_model(config)

model.trainable = False

x, y = next(iter(dataloader))
print(model.summary())
print(x.shape, y.shape)
print(config)



his = model.fit(dataloader, epochs=config.trainer.epochs)
print(his)
