
import numpy as np
from eyenet import CHASE_DB1
from eyenet import UNet
from eyenet import get_configured


class TestInference:
    def test_prediction_output(self):
        cfg = get_configured('eyenet/cfg/chase_db1.yml')
        db = CHASE_DB1(cfg).load()
        x,y = next(iter(db))

        model = UNet(
            backbone='efficientnetb0',
            input_size=224,
            num_classes=1,
            activation='sigmoid',
        )
        y_pred = model(x, training=True)

        assert y_pred.shape[-1] == y.shape[-1]
        assert y_pred.shape[0] == cfg.dataset.batch_size
        assert np.all(y_pred >= 0) and np.all(y_pred <= 1), "Model output is not in range [0, 1]"


