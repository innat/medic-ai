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
from .utils import get_configured
from .dataloader import get_dataloader
from pathlib import Path
from omegaconf import OmegaConf
from tensorflow import keras

config_path = "eyenet/cfg/default.yml"
config = get_configured(config_path=config_path)
dataloader = get_dataloader(config)
model = get_model(config)
model.trainable = False

x, y = next(iter(dataloader))
print(model.summary())
print(x.shape, y.shape)
print(config)
his = model.fit(dataloader, epochs=config.trainer.epochs)
print(his)
