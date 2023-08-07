import pytest

from eyenet.utils import MasterConfigurator
from eyenet.dataloader import CHASE_DB1
from eyenet.nets import UNet


def get_configured(config_path):
    config = MasterConfigurator(config_path)
    return config.get_seg_cfg()


@pytest.mark.parametrize(
    "config_path, db_class, model_config, output_shape",
    [
        (
            "eyenet/cfg/chase_db1.yml",
            CHASE_DB1,
            {
                "model_class": UNet,
                "backbone": "efficientnetb0",
                "input_size": 224,
                "num_classes": 1,
                "activation": "sigmoid",
            },
            (224, 224, 1),  # adjust this based on your expected output shape
        ),
    ],
)
class TestInference:
    def test_prediction_output(self, config_path, db_class, model_config, output_shape):
        cfg = get_configured(config_path)
        db = db_class(cfg).load()
        x, y = next(iter(db))

        model_class = model_config.pop("model_class")
        model = model_class(**model_config)
        y_pred = model(x, training=True)

        assert y_pred.shape[-1] == y.shape[-1]
        assert y_pred.shape[0] == cfg.dataset.batch_size
