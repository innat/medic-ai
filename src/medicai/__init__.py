import os
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from medicai import nets
from medicai import dataset
from medicai import utils

__all__ = [
    "nets",
    "dataset",
    "utils",
]

__version__ = "0.0.1"
