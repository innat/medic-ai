__version__ = "0.0.1"

import os
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from eyenet.nets import DuelAttentionNet, UNet
from eyenet.utils import get_configured
from eyenet.dataloader import APTOSDataloader, CHASE_DB1

__all__ = ["DuelAttentionNet", "UNet", "get_configured", "APTOSDataloader" "CHASE_DB1"]
