
from .net import get_model
from .utils import get_config
from .data import get_dataloader
from pathlib import Path
from omegaconf import OmegaConf

config_path='./cfg/default.yml'
config = get_config(config=config_path)

project_path = Path(config.project.path) / config.model.name / config.dataset.name / "run"
(project_path / 'weights').mkdir(parent=True, exist_ok=True)
(project_path / 'images').mkdir(parent=True, exist_ok=True)
config.project.path = str(project_path)
(project_path / "config.yaml").write_text(OmegaConf.to_yaml(config))
(project_path / "config_original.yaml").write_text(OmegaConf.to_yaml(config_path))

print(config)

dataloader = get_dataloader(config)
model = get_model(config)

