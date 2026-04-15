import pytest

from medicai.utils.registry import ModelRegistry, TabularOutput


class DummyBackbone:
    def __init__(self, width=16):
        self.width = width


class DummySegmentor:
    ALLOWED_BACKBONE_FAMILIES = ["resnet", "densenet"]

    def __init__(self, depth=3):
        self.depth = depth


@pytest.mark.unit
def test_registry_register_and_get_create():
    registry = ModelRegistry()

    registry.register(name="resnet18", family="resnet")(DummyBackbone)
    registry.register(name="unet", type="segmentation")(DummySegmentor)

    entry = registry.get_entry("resnet18")
    assert entry["class"] is DummyBackbone
    assert entry["family"] == ["resnet"]

    segmentor_entry = registry.get_entry("UNET")
    assert segmentor_entry["class"] is DummySegmentor
    assert segmentor_entry["allowed_families"] == ["resnet", "densenet"]

    instance = registry.create("resnet18", width=32)
    assert isinstance(instance, DummyBackbone)
    assert instance.width == 32


@pytest.mark.unit
def test_registry_duplicate_registration_raises():
    registry = ModelRegistry()
    registry.register(name="resnet18", family="resnet")(DummyBackbone)

    with pytest.raises(KeyError, match="already registered"):
        registry.register(name="resnet18", family="resnet")(DummyBackbone)


@pytest.mark.unit
def test_registry_unknown_type_raises():
    registry = ModelRegistry()
    with pytest.raises(ValueError, match="Unknown model type"):
        registry.register(name="x", type="classifier")(DummyBackbone)


@pytest.mark.unit
def test_registry_unknown_model_raises_with_available_names():
    registry = ModelRegistry()
    registry.register(name="resnet18", family="resnet")(DummyBackbone)

    with pytest.raises(KeyError, match="Available models"):
        registry.get_entry("does-not-exist")


@pytest.mark.unit
def test_registry_list_and_tabular_output_repr():
    registry = ModelRegistry()
    registry.register(name="resnet18", family="resnet")(DummyBackbone)
    registry.register(name="unet", type="segmentation")(DummySegmentor)

    output = registry.list()
    assert isinstance(output, TabularOutput)
    assert isinstance(output._repr_html_(), str)
