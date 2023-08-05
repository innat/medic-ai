import click
import os
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from .nets import get_model
from .utils import get_configured
from .dataloader import get_dataloader


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config",
    type=str,
    default="eyenet/cfg/default.yml",
    help="config for training",
)
@click.option(
    "--task-type",
    type=click.Choice(("classification", "segmentation")),
    default="classification",
    help="task type for training",
)
def train(config: str, task_type: str):
    cfg = get_configured(config)
    db = get_dataloader(cfg)
    model = get_model(cfg)
    hist = model.fit(db)
    return hist


@cli.command()
@click.option(
    "--image-path",
    type=click.Path(exists=True),
    required=True,
    help="path of image for inference",
)
@click.option(
    "--config",
    type=str,
    default="eyenet/cfg/default.yml",
    help="config for training",
)
def inference(image_path: str, config: str):
    cfg = get_configured(config)
    db = get_dataloader(cfg)
    model = get_model(cfg)
    y_pred = model.predict(db)
    return y_pred


if __name__ == "__main__":
    cli()
