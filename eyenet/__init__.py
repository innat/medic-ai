
from .net import classifier
from .utils import get_config
from .data import create_dataset
from pathlib import Path



config = get_config()
project_path = Path(config.project.path) / config.model.name / config.dataset.name

dataloader = create_dataset(config)
