import click
import os
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from medicai import nets
from medicai import dataloader
from medicai import utils


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config",
    type=str,
    default="medicai/cfg/default.yml",
    help="config for training",
)
@click.option(
    "--task-type",
    type=click.Choice(("classification", "segmentation")),
    default="classification",
    help="task type for training",
)
def train(config: str, task_type: str):
    master_cfg = utils.MasterConfigurator("medicai/cfg/aptos.yml")
    cls_cfg = master_cfg.get_cls_cfg(
        model_name="efficientnetb0",
        input_size=224,
        num_classes=5,
        metrics="cohen_kappa",
        losses="cohen_kappa",
    )
    data = dataloader.APTOSDataloader(cls_cfg)
    model = nets.DuelAttentionNet(cls_cfg)
    hist = model.fit(data.load())
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
    default="medicai/cfg/default.yml",
    help="config for training",
)
def inference(image_path: str, config: str):
    master_cfg = utils.MasterConfigurator("medicai/cfg/aptos.yml")
    cls_cfg = master_cfg.get_cls_cfg(
        model_name="efficientnetb0",
        input_size=224,
        num_classes=5,
        metrics="cohen_kappa",
        losses="cohen_kappa",
    )
    data = dataloader.APTOSDataloader(cls_cfg)
    model = nets.DuelAttentionNet(cls_cfg)
    y_pred = model.predict(data)
    return y_pred


if __name__ == "__main__":
    cli()
