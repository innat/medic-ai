import pytest

from src.medicai.utils import MasterConfigurator


def get_cls_config(config_path, **kwargs):
    config = MasterConfigurator(config_path)
    return config.get_cls_cfg(**kwargs)


def get_seg_config(config_path, **kwargs):
    config = MasterConfigurator(config_path)
    return config.get_seg_cfg(**kwargs)


@pytest.mark.parametrize(
    "get_config_func, config_args, model_name, backbone, image_size",
    [
        (
            get_cls_config,
            {
                "config_path": "eyenet/cfg/aptos.yml",
                "model_name": "efficientnetb0",
                "input_size": 100,
                "num_classes": 5,
                "metrics": "cohen_kappa",
                "losses": "cohen_kappa",
            },
            "efficientnetb0",
            None,
            100,
        ),
        (
            get_seg_config,
            {
                "config_path": "eyenet/cfg/chase_db1.yml",
                "model_name": "unet",
                "backbone": "efficientnetb0",
                "input_size": 100,
                "num_classes": 1,
                "metrics": "accuracy",
                "losses": "binary_crossentropy",
            },
            "unet",
            "efficientnetb0",
            100,
        ),
    ],
)
def test_config(get_config_func, config_args, model_name, backbone, image_size):
    config = get_config_func(**config_args)

    assert config.model.name == model_name
    assert config.model.backbone == backbone
    assert config.dataset.image_size >= image_size
