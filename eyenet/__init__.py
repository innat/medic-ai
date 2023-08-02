import os
import warnings

__version__ = "0.0.1"

def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from .nets import get_model
from .utils import get_configured
from .dataloader import get_dataloader
from pathlib import Path
from omegaconf import OmegaConf
from tensorflow import keras

import click

@click.group()
def cli():
    pass

config_path = "eyenet/cfg/default.yml"
config = get_configured(config_path=config_path)
dataloader = get_dataloader(config)
model = get_model(config)
model.trainable = False
x, y = next(iter(dataloader))
print(model.summary())
print(x.shape, y.shape)
print(config)
his = model.fit(dataloader, epochs=config.trainer.epochs)
print(his)

@cli.command()
@click.option(
    "--data-config",
    type=str, 
    default='',
    help='The config path of target dataset.'
)
@click.option(
    "--model-config",
    type=str,
    default='',
    help='The config path of the target model.',
)
@click.option(
    "--task-type",
    type=click.Choice(('classification', 'segmentation')),
    default='classification',
)
@click.option(
    "devices",
    type=click.Choice(('cpu', 'gpu', 'tpu')),
    default='gpu',
    help='The device for training.',
)
def train(
    data_config:str, model_config:str, task_type:str, devices:str
):
    pass

@cli.command()
@click.option(
    "--image-path",
    type=click.Path(exists=True), 
    required=True,
    help='An image for inference.',
)
@click.option(
    "--config-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--save-path",
    type=click.Path(exists=True, dir_okay=True),
    defualt=None,
)
@click.option(
    "devices",
    type=click.Choice(('cpu', 'gpu', 'tpu')),
    default='gpu',
    help='The device for inference.',
)
def inference(
    image_path:str, config_path:str, save_path:str, devices:str
):
    pass

if __name__ == '__main__':
    cli()