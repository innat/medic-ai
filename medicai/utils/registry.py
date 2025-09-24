class BackboneFactoryRegistry:
    """A factory registry for backbone models.

    This class provides a centralized system for registering, retrieving, and
    instantiating backbone models, such as ResNet, DenseNet, etc. It uses a
    decorator-based approach to automatically register models when their
    respective modules are imported. This simplifies model management and
    allows for easy discovery of available backbones.
    """

    def __init__(self):
        self._registry = {}

    def register(
        self,
        name=None,
        family=None,
    ):
        """Registers a backbone model using a decorator.

        This method is designed to be used as a decorator on a class. It adds the
        class to the registry with its name and family. The `name` defaults to
        the class name if not provided. The `family` argument helps in categorizing
        backbones and can be a single string or a list of strings.

        Args:
            name: (Optional) The name to register the class under. Defaults to
                the class name.
            family: The family of the backbone, e.g., 'resnet' or ['densenet', 'resnet'].

        Returns:
            A decorator that registers the class.

        Example:
            >>> @registration.register(name="resnet18", family="resnet")
            ... class ResNet18:
            ...     pass
        """

        def decorator(cls):
            key = (name or cls.__name__).lower()
            fam = [family] if isinstance(family, str) else family
            entry = {"class": cls, "family": [f.lower() for f in (fam or [])]}
            self._registry[key] = entry
            return cls

        return decorator

    def get_entry(self, name: str):
        """Retrieves a backbone's registry entry by its name.

        Args:
            name: The name of the backbone to retrieve (case-insensitive).

        Returns:
            A dictionary containing the class and family of the backbone.

        Raises:
            KeyError: If the specified backbone is not found in the registry.
        """
        key = name.lower()
        if key not in self._registry:
            raise KeyError(f"Backbone '{name}' not found. Available: {list(self._registry.keys())}")
        return self._registry[key]

    def get(self, name: str):
        """Retrieves a backbone's class by its name.

        Args:
            name: The name of the backbone to retrieve (case-insensitive).

        Returns:
            The Python class object for the specified backbone.
        """
        return self.get_entry(name)["class"]

    def create(self, name, **kwargs):
        """Instantiates a backbone model from the registry.

        Args:
            name: The name of the backbone to instantiate (case-insensitive).
            **kwargs: Keyword arguments to be passed to the class constructor.

        Returns:
            An instance of the specified backbone class.
        """
        cls = self.get(name)
        return cls(**kwargs)

    def list(self, family=None):
        """Lists registered backbones.

        Args:
            family: (Optional) Filters the list to a specific family or families.

        Returns:
            A list of backbone names or full registry entries.
        """

        def list(cls, family=None):
            # Only pass family model
            if family is not None:
                names = [
                    name
                    for name, entry in cls._registry.items()
                    if (entry["family"] == family)
                    or (isinstance(entry["family"], (list, tuple)) and family in entry["family"])
                ]
                # Same pretty style as full listing
                lines = ["Family   - Variants", "-" * 20]
                variants = ", ".join(names) if names else "None"
                lines.append(f"{family:<8} - {variants}")
                return "\n".join(lines)

            # Group by family
            grouped = {}
            for name, entry in cls._registry.items():
                fam = entry["family"]
                if isinstance(fam, (list, tuple)):
                    for f in fam:
                        grouped.setdefault(f, []).append(name)
                else:
                    grouped.setdefault(fam, []).append(name)

            # Pretty table
            lines = ["Family   - Variants", "-" * 20]
            for fam, models in grouped.items():
                variants = ", ".join(models)
                lines.append(f"{fam:<8} - {variants}")
            return "\n".join(lines)


registration = BackboneFactoryRegistry()
