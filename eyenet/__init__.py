import os
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from .net import get_model
from .utils import get_config
from .data import get_dataloader
from pathlib import Path
from omegaconf import OmegaConf
from tensorflow import keras
import tensorflow_addons as tfa

config_path = "eyenet/cfg/default.yml"
config = get_config(config_path=config_path)

project_path = Path(config.project.path) / config.dataset.name / config.model.name / "run"
(project_path / "weights").mkdir(parents=True, exist_ok=True)
(project_path / "images").mkdir(parents=True, exist_ok=True)
config.project.path = str(project_path)
# (project_path / "config.yaml").write_text(OmegaConf.to_yaml(config))
# (project_path / "config_original.yaml").write_text(OmegaConf.to_yaml(config_path))

dataloader = get_dataloader(config)
model = get_model(config)

x, y = next(iter(dataloader))
print(model.summary())
print(x.shape, y.shape)
print(config)

if config.losses.primary == "cohen_kappa_loss":
    primary_loss = tfa.losses.WeightedKappaLoss(
        num_classes=config.dataset.num_classes,
        weightage="quadratic",
        name="primary_loss",
    )
if config.losses.auxilary == "categorical_crossentropy":
    auxilary_loss = keras.losses.CategoricalCrossentropy(
        label_smoothing=config.losses.label_smoothing, name="aux_loss"
    )

if config.metrics.primary == "cohen_kappa":
    primary_metrics = tfa.metrics.CohenKappa(
        num_classes=config.dataset.num_classes,
        weightage="quadratic",
        name="primary_metrics",
    )
if config.metrics.auxilary == "accuracy":
    auxilary_metrics = keras.metrics.CategoricalAccuracy(name="auxilary_metrics")

if config.trainer.optimizer == "adam":
    optim = keras.optimizers.Adam(learning_rate=config.trainer.learning_rate)

model.compile(
    loss={
        "primary": primary_loss,
        "auxilary": auxilary_loss,
    },
    metrics={
        "primary": [
            "accuracy",
            primary_metrics,
        ],
        "auxilary": [auxilary_metrics],
    },
    loss_weights={"primary": 1.0, "auxilary": 0.3},
    optimizer=optim,
)

model.trainable = False
his = model.fit(dataloader, epochs=config.trainer.epochs)
print(his)
