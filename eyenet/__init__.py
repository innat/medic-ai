
__version__ = "0.0.1"

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

__all__ = ['get_model', 'get_configured', 'get_dataloader']
