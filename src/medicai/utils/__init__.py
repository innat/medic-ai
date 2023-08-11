from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from .grad_accumulator import GradientAccumulator


class Configurator:
    def __init__(self, config_path: Path) -> None:
        config = OmegaConf.load(config_path)
        config_original: DictConfig = config.copy()
        self.config_original = config_original
        self.config = config
    
    def setup(self):
        project_path = Path(self.config.project.path) / self.config.dataset.name / self.config.model.name / "run"
        (project_path / "weights").mkdir(parents=True, exist_ok=True)
        (project_path / "images").mkdir(parents=True, exist_ok=True)
        self.config.project.path = str(project_path)


    def update_cls_cfg(self, model_name=None, input_size=None, num_classes=None):

        valid_values = {
            "model_name": ['efficientnetb0', 'efficientnetb1', 'resnet50'],
        }

        model_name = model_name or self.config.model.name
        input_size = input_size or self.config.dataset.image_size
        num_classes = num_classes or self.config.dataset.num_classes
            
        params = {
            "model_name": model_name,
        }
        for param_name, value in params.items():
            self._validate_param(param_name, value, valid_values[param_name])

        self.config.model.name = model_name
        self.config.dataset.image_size = input_size
        self.config.dataset.num_classes = num_classes

        return self.config

    def update_seg_cfg(self, model_name=None, backbone=None, input_size=None, num_classes=None, shuffle=False):
        
        valid_values = {
            "model_name": ['unet'],
            "backbone": ["efficientnetb0"],
        }

        model_name = model_name or self.config.model.name
        backbone = backbone or self.config.model.backbone
        input_size = input_size or self.config.dataset.image_size
        num_classes = num_classes or self.config.dataset.num_classes
        shuffle = shuffle or self.config.dataset.shuffle
        
        params = {
            "model_name": model_name,
            "backbone": backbone
        }
        for param_name, value in params.items():
            self._validate_param(param_name, value, valid_values[param_name])

        self.config.model.name = model_name
        self.config.model.backbone = backbone
        self.config.dataset.image_size = input_size
        self.config.dataset.num_classes = num_classes
        self.config.dataset.shuffle = shuffle

        return self.config
    
    def _validate_param(self, name, value, valid_values_list):
        if value not in valid_values_list:
            valid_str = ', '.join(valid_values_list)
            raise ValueError(
                f"The {name} '{value}' is not supported. Allowed values are: {valid_str}."
            )
