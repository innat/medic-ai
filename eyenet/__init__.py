__version__ = "0.0.1"

import os
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from eyenet import nets
from eyenet import dataloader
from eyenet import utils

__all__ = [
    "nets",
    "dataloader",
    "utils",
]
