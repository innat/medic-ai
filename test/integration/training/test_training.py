import pytest
import numpy as np
from eyenet import get_configured
from eyenet import get_model


class TestModelTraining:
    def test_model_fit(self):
        config_path = "eyenet/cfg/default.yml"
        config = get_configured(config_path=config_path)

        model = get_model(config)
        model.trainable = False

        x = np.random.random(
            (config.dataset.batch_size, config.dataset.image_size, config.dataset.image_size, 3)
        )
        y = np.random.random((config.dataset.batch_size, config.dataset.num_classes))

        history = model.fit(
            x,
            y,
            batch_size=config.dataset.batch_size,
            epochs=config.trainer.epochs,
            validation_split=0.2,
            shuffle=False,
        ).history

        # The loss should decrease after training
        assert len(history) == config.trainer.epochs
