from omegaconf import DictConfig, ListConfig, OmegaConf
from pathlib import Path


def get_config(config_path: Path):
    config = OmegaConf.load(config_path)
    config_original: DictConfig = config.copy()
    return config
