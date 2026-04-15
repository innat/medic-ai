import pytest

from medicai.utils.general import BaseEnum, camel_to_snake


class DummyEnum(BaseEnum):
    FOO = "foo"
    BAR = "bar"


@pytest.mark.unit
@pytest.mark.parametrize(
    "value,expected",
    [
        ("BinaryDiceLoss", "binary_dice_loss"),
        ("UNETRPlusPlus", "unetr_plus_plus"),
        ("DiceCELoss", "dice_ce_loss"),
        ("A1B2", "a1_b2"),
    ],
)
def test_camel_to_snake(value, expected):
    assert camel_to_snake(value) == expected


@pytest.mark.unit
def test_base_enum_helpers():
    assert DummyEnum.to_dict() == {"FOO": "foo", "BAR": "bar"}
    assert DummyEnum.values() == ["foo", "bar"]
