
from typing import Union, Optional, List

class BackboneFactoryRegistry:
    def __init__(self):
        self._registry = {}

    def register(
        self,
        name: Optional[str] = None,
        family: Union[str, List[str]] = None,
        aliases: Optional[List[str]] = None,
    ):
        """Register a backbone model."""
        def decorator(cls):
            key = (name or cls.__name__).lower()
            fam = [family] if isinstance(family, str) else family
            entry = {"class": cls, "family": [f.lower() for f in (fam or [])]}
            self._registry[key] = entry

            if aliases:
                for alias in aliases:
                    self._registry[alias.lower()] = entry

            return cls

        return decorator  # <-- important to return the decorator

    def get_entry(self, name: str):
        key = name.lower()
        if key not in self._registry:
            raise KeyError(
                f"Backbone '{name}' not found. Available: {list(self._registry.keys())}"
            )
        return self._registry[key]

    def get(self, name: str):
        return self.get_entry(name)["class"]

    def create(self, name: str, **kwargs):
        cls = self.get(name)
        return cls(**kwargs)

    def list(self, family: Union[str, List[str]] = None, details: bool = False):
        families = [family] if isinstance(family, str) else family
        result = []
        for k, v in self._registry.items():
            if not families or any(f in v["family"] for f in families):
                result.append((k, v) if details else k)
        return result