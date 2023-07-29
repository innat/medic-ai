from omegaconf import DictConfig, OmegaConf
from pathlib import Path


def get_config(config_path: Path):
    config = OmegaConf.load(config_path)
    config_original: DictConfig = config.copy()  # noqa: F841
    return config
