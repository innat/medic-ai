from omegaconf import DictConfig, OmegaConf
from pathlib import Path


def get_config(config_path: Path):
    config = OmegaConf.load(config_path)
    config_original: DictConfig = config.copy()  # noqa: F841

    if config.dataset.name not in ("aptos", "eyepacks"):
        raise ValueError(
            "Supported data sets are aptos and eyepacks ",
            f"Got: {config.dataset.name}",
        )

    if config.model.name != "efficientnet":
        raise ValueError(
            "Supported backbone model is efficientnet ",
            f"Got: {config.model.name}",
        )
    elif config.model.layers != "block5a_expand_conv":
        raise ValueError(
            "Supported intermediate layer of efficientnet is  block5a_expand_conv ",
            f"Got: {config.model.layers}",
        )

    if config.model.weight not in ("imagenet", None):
        raise ValueError(
            "Supported weight can be imagenet or None ",
            f"Got: {config.model.weight}",
        )

    return config
