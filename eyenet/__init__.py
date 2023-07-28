
from .net import get_model
from .utils import get_config
from .data import get_dataloader
from pathlib import Path


config = get_config()
dataloader = get_dataloader(config)
model = get_model(config)

project_path = Path(config.project.path) / config.model.name / config.dataset.name

