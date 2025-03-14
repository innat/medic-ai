import os
import warnings


def warn(*args, **kwargs):
    pass

warnings.warn = warn
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# from medicai import datasets, models, utils

__all__ = [
    "models",
    "transforms",
    "datasets",
    "utils",
]

__version__ = "0.0.1"
