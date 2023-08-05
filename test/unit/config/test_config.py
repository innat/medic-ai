import pytest

import sys
import os
from eyenet import get_configured


class TestConfig:
    def test_get_model_name(self):
        backbone = "efficientnet"
        config_path = "eyenet/cfg/default.yml"
        config = get_configured(config_path=config_path)
        assert config.model.name == backbone

    def test_get_dataset_name(self):
        db_name = "aptos"
        config_path = "eyenet/cfg/default.yml"
        config = get_configured(config_path=config_path)
        assert config.dataset.name == db_name
