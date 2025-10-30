import re
from enum import Enum


def camel_to_snake(name: str) -> str:
    # Step 1: Put underscore between lower-uppercase or digit-uppercase
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    # Step 2: Handle acronym + word boundary (e.g., "CE" + "Loss")
    s2 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s1)
    return s2.lower()


class EnumMixin:
    @classmethod
    def to_dict(cls):
        # Returns a dictionary mapping member names to their values.
        return {item.name: item.value for item in cls}

    @classmethod
    def values(cls):
        # Returns a list of all member string values.
        return [item.value for item in cls]


class BaseEnum(EnumMixin, str, Enum): ...
