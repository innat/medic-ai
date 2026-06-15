import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KERAS_BACKEND", "tensorflow")


class _FakeKerasObject:
    """Small import-time stand-in for Keras objects used by autodoc."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return args[0] if args else self


def _identity_keras_decorator(*args, **kwargs):
    def decorator(obj):
        return obj

    return decorator


class _FakeKerasModule(types.ModuleType):
    def __getattr__(self, name):
        fake = type(name, (_FakeKerasObject,), {})
        setattr(self, name, fake)
        return fake


def _install_fake_keras():
    def keras_type(name, bases=(_FakeKerasObject,)):
        return type(name, bases, {"__module__": "keras"})

    keras = _FakeKerasModule("keras")
    layers = _FakeKerasModule("keras.layers")
    metrics = _FakeKerasModule("keras.metrics")
    losses = _FakeKerasModule("keras.losses")
    initializers = _FakeKerasModule("keras.initializers")
    constraints = _FakeKerasModule("keras.constraints")
    regularizers = _FakeKerasModule("keras.regularizers")
    activations = _FakeKerasModule("keras.activations")
    callbacks = _FakeKerasModule("keras.callbacks")
    random = _FakeKerasModule("keras.random")
    saving = types.ModuleType("keras.saving")
    utils = types.ModuleType("keras.utils")
    backend = types.ModuleType("keras.backend")
    config = types.ModuleType("keras.config")
    ops = _FakeKerasModule("keras.ops")

    keras.Model = keras_type("Model")
    keras.Sequential = keras_type("Sequential")
    keras.Input = lambda *args, **kwargs: _FakeKerasObject()
    layers.Layer = keras_type("Layer")
    metrics.Metric = keras_type("Metric")
    losses.Loss = keras_type("Loss")
    initializers.Initializer = keras_type("Initializer")
    constraints.Constraint = keras_type("Constraint")
    regularizers.Regularizer = keras_type("Regularizer")
    callbacks.Callback = keras_type("Callback")
    activations.get = lambda activation=None: activation
    activations.serialize = lambda activation=None: activation
    random.SeedGenerator = keras_type("SeedGenerator")
    random.uniform = lambda *args, **kwargs: _FakeKerasObject()
    saving.register_keras_serializable = _identity_keras_decorator
    saving.serialize_keras_object = lambda obj: obj
    utils.register_keras_serializable = _identity_keras_decorator
    utils.PyDataset = keras_type("PyDataset")
    utils.to_categorical = lambda *args, **kwargs: None
    backend.epsilon = lambda: 1e-7
    backend.is_keras_tensor = lambda tensor: False
    config.backend = lambda: os.environ.get("KERAS_BACKEND", "tensorflow")
    config.epsilon = lambda: 1e-7

    keras.layers = layers
    keras.metrics = metrics
    keras.losses = losses
    keras.initializers = initializers
    keras.constraints = constraints
    keras.regularizers = regularizers
    keras.activations = activations
    keras.callbacks = callbacks
    keras.random = random
    keras.saving = saving
    keras.utils = utils
    keras.backend = backend
    keras.config = config
    keras.ops = ops

    for module in (
        keras,
        layers,
        metrics,
        losses,
        initializers,
        constraints,
        regularizers,
        activations,
        callbacks,
        random,
        saving,
        utils,
        backend,
        config,
        ops,
    ):
        module.__file__ = str(Path(__file__).resolve())
        sys.modules.setdefault(module.__name__, module)


_install_fake_keras()

project = "medic-ai"
copyright = "2024, Innat"
author = "Innat"

from medicai import __version__

release = __version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "notfound.extension",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}

root_doc = "index"
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "guides/data-preparation.md",
]

autosummary_generate = True
autodoc_default_options = {
    "inherited-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False

autodoc_mock_imports = [
    "cv2",
    "jax",
    "tensorflow",
    "torch",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    # Keras and TensorFlow do not currently expose Sphinx inventories at these
    # documentation roots, so keep the official bases with an empty local
    # inventory to avoid warnings-as-errors failures.
    "keras": ("https://keras.io/api/", "_static/empty-objects.inv"),
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python/",
        "_static/empty-objects.inv",
    ),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
}

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "medic-ai"
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")
html_theme_options = {
    "github_url": "https://github.com/innat/medic-ai",
    "navbar_align": "left",
    "header_dropdown_text": "Misc",
    "header_links_before_dropdown": 4,
    "show_nav_level": 2,
    "show_version_warning_banner": True,
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/medicai/",
            "icon": "fa-brands fa-python",
        },
    ],
    "navbar_start": [],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["search-field", "theme-switcher", "navbar-icon-links"],
    "navbar_persistent": [],
}

html_context = {
    "github_user": "innat",
    "github_repo": "medic-ai",
    "github_version": "main",
    "doc_path": "docs",
}

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "fieldlist",
    "substitution",
]
myst_heading_anchors = 3

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

nitpicky = True
suppress_warnings = [
    "docutils",
    "ref",
]


def _format_keras_bases(app, name, obj, options, bases):
    for index, base in enumerate(bases):
        if getattr(base, "__module__", None) == "keras":
            bases[index] = f"``keras.{base.__qualname__}``"


def setup(app):
    app.connect("autodoc-process-bases", _format_keras_bases)
