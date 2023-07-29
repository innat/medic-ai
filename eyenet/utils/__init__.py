from omegaconf import DictConfig, OmegaConf
from pathlib import Path


def get_configured(config_path: Path):
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
    elif config.model.layers[0] != "block5a_expand_conv":
        raise ValueError(
            "Supported intermediate layer of efficientnet is  block5a_expand_conv ",
            f"Got: {config.model.layers}",
        )

    if config.model.weight not in ("imagenet", None):
        raise ValueError(
            "Supported weight can be imagenet or None ",
            f"Got: {config.model.weight}",
        )

    if config.losses.primary not in ("categorical_crossentropy", "cohen_kappa_loss"):
        raise ValueError("not supported")

    if config.losses.auxilary not in ("categorical_crossentropy", "cohen_kappa_loss"):
        raise ValueError("not supported")

    if config.metrics.primary not in ("cohen_kappa", "accuracy"):
        raise ValueError("not supported")

    if config.metrics.auxilary not in ("cohen_kappa", "accuracy"):
        raise ValueError("not supported")

    if config.trainer.optimizer != "adam":
        raise ValueError("not supported")


    project_path = Path(config.project.path) / config.dataset.name / config.model.name / "run"
    (project_path / "weights").mkdir(parents=True, exist_ok=True)
    (project_path / "images").mkdir(parents=True, exist_ok=True)
    config.project.path = str(project_path)

    # (project_path / "config.yaml").write_text(OmegaConf.to_yaml(config))
    # (project_path / "config_original.yaml").write_text(OmegaConf.to_yaml(config_path))

    return config
