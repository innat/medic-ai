from tabulate import tabulate


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
        """Lists registered backbones, optionally filtered by family, and returns a
        nicely formatted string using the tabulate library.

        Args:
            family: (Optional) A single family name (e.g., 'resnet') to filter
                    the list. If not specified, all backbones are listed.

        Returns:
            A string containing a formatted table of backbone names.
        """
        headers = ["Family", "Variants"]

        if family is not None:
            names = [name for name, entry in self._registry.items() if family in entry["family"]]
            table_data = [(family, ", ".join(names) if names else "None")]
        else:
            grouped = {}
            for name, entry in self._registry.items():
                for fam in entry["family"]:
                    grouped.setdefault(fam, []).append(name)

            table_data = [(fam, ", ".join(models)) for fam, models in sorted(grouped.items())]

        return tabulate(table_data, headers=headers, tablefmt="mixed_outline")


registration = BackboneFactoryRegistry()
