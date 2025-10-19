import logging
import os
import warnings

# Disable Tensorflow logging.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Disable Python warnings
warnings.warn = lambda *args, **kwargs: None
warnings.simplefilter(action="ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.disable(logging.WARNING)
logging.getLogger("tensorflow").disabled = True

from medicai import dataloader, models, utils

__all__ = [
    "models",
    "transforms",
    "dataloader",
    "utils",
]

__version__ = "0.0.3"
