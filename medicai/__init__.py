import importlib

__version__ = "0.0.3"
__all__ = [
    "models",
    "transforms",
    "layers",
    "metrics",
    "losses",
    "dataloader",
    "utils",
]


def __getattr__(name):
    """
    Lazily import submodules when accessed as attributes.
    Example:
        import medicai
        medicai.losses  -> loads medicai.losses only when needed
    """
    if name in __all__:
        module = importlib.import_module(f"medicai.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'medicai' has no attribute '{name}'")


def version():
    return __version__
