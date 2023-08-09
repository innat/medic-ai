import pytest
import numpy as np
import pandas as pd

from medic.utils import MasterConfigurator
from medic.nets import UNet


@pytest.mark.parametrize("config_path", ["eyenet/cfg/chase_db1.yml"])
class TestModelTraining:
    def test_model_fit(self, config_path):
        config = MasterConfigurator(config_path=config_path).get_seg_cfg(
            model_name="unet",
            backbone="efficientnetb0",
            input_size=100,
            num_classes=1,
            metrics="accuracy",
            losses="binary_crossentropy",
        )

        model = UNet(config)
        model.trainable = False

        x = np.random.random(
            (config.dataset.batch_size, config.dataset.image_size, config.dataset.image_size, 3)
        )
        y = np.random.random(
            (
                config.dataset.batch_size,
                config.dataset.image_size,
                config.dataset.image_size,
                config.dataset.num_classes,
            )
        )

        history = model.fit(
            x,
            y,
            batch_size=config.dataset.batch_size,
            epochs=config.trainer.epochs,
            validation_split=0.2,
            shuffle=False,
        ).history

        # The loss should decrease after training
        assert len(pd.DataFrame(history)) == config.trainer.epochs
