from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from .grad_accumulator import GradientAccumulator


valid_values = {
    "model_name": ['efficientnetb0', 'efficientnetb1', 'resnet50'],
    "metrics": ['cohen_kappa'],
    "losses": ['cohen_kappa', 'mse']
}

class Configurator:
    def __init__(self, config_path: Path) -> None:
        config = OmegaConf.load(config_path)
        config_original: DictConfig = config.copy()

        project_path = Path(config.project.path) / config.dataset.name / config.model.name / "run"
        (project_path / "weights").mkdir(parents=True, exist_ok=True)
        (project_path / "images").mkdir(parents=True, exist_ok=True)
        config.project.path = str(project_path)

        if config.dataset.name not in ("aptos", "chase_db1"):
            raise ValueError(
                "Supported data sets are aptos and chase_db1 ",
                f"Got: {config.dataset.name}",
            )

        self.config_original = config_original
        self.config = config

    def update_cls_cfg(self, model_name, input_size, num_classes, metrics, losses, **kwargs):
        model_name = model_name or self.config.model.name
        input_size = input_size or self.config.dataset.image_size
        num_classes = num_classes or self.config.dataset.num_classes
        metrics = metrics or self.config.metrics
        losses = losses or self.config.losses
            
        params = {
            "model_name": model_name,
            "metrics": metrics,
            "losses": losses
        }

        for param_name, value in params.items():
            self._validate_param(param_name, value, valid_values[param_name])

        self.config.model.name = model_name
        self.config.dataset.image_size = input_size
        self.config.dataset.num_classes = num_classes
        self.config.metrics = metrics
        self.config.losses = losses

        return self.config

    def update_seg_cfg(self, **kwargs):
        model_name = kwargs.get("model_name", self.config.model.name)
        backbone = kwargs.get("backbone", self.config.model.backbone)
        input_size = kwargs.get("image_size", self.config.dataset.image_size)
        num_classes = kwargs.get("num_classes", self.config.dataset.num_classes)
        metrics = kwargs.get("metrics", self.config.metrics)
        losses = kwargs.get("losses", self.config.losses)

        if model_name != self.config.model.name:
            raise ValueError(
                "Supported model is UNet ",
                f"Got: {self.config.model.name}",
            )

        if backbone != self.config.model.backbone:
            raise ValueError(
                "Supported backbone model of UNet is efficientnetb0 ",
                f"Got: {self.config.model.backbone}",
            )

        if metrics != "accuracy":
            raise ValueError(
                "Supported metrics is accuracy ",
                f"Got: {metrics}",
            )

        if losses != "binary_crossentropy":
            raise ValueError(
                "Supported metrics is binary_crossentropy ",
                f"Got: {losses}",
            )

        self.config.model.name = model_name
        self.config.model.backbone = backbone
        self.config.dataset.image_size = input_size
        self.config.dataset.num_classes = num_classes
        self.config.metrics = metrics
        self.config.losses = losses

        return self.config
    
    def _validate_param(self, name, value, valid_values_list):
        if value not in valid_values_list:
            valid_str = ', '.join(valid_values_list)
            raise ValueError(
                f"Supported {name} are {valid_str}. Got: {value}"
            )
