from rich.console import Console
from rich.table import Table


class TabularOutput:
    def __init__(self, table):
        self.table = table

    def _repr_html_(self):
        # Using a Console to capture the rendered table as HTML
        console = Console(record=True, force_terminal=False)
        return console.export_html(inline_styles=True)

    # Keeping `__repr__` for non-Jupyter environments (like simple Python console)
    def __repr__(self):
        console = Console()
        # This will still print the text version once if not in Jupyter,
        # or if explicitly called. In Jupyter, _repr_html_ takes priority.
        console.print(self.table)
        return ""


# class BackboneFactoryRegistry:
#     """A factory registry for backbone models.

#     This class provides a centralized system for registering, retrieving, and
#     instantiating backbone models, such as ResNet, DenseNet, etc. It uses a
#     decorator-based approach to automatically register models when their
#     respective modules are imported. This simplifies model management and
#     allows for easy discovery of available backbones.
#     """

#     def __init__(self):
#         self._registry = {}

#     def register(
#         self,
#         name=None,
#         family=None,
#     ):
#         """Registers a backbone model using a decorator.

#         This method is designed to be used as a decorator on a class. It adds the
#         class to the registry with its name and family. The `name` defaults to
#         the class name if not provided. The `family` argument helps in categorizing
#         backbones and can be a single string or a list of strings.

#         Args:
#             name: (Optional) The name to register the class under. Defaults to
#                 the class name.
#             family: The family of the backbone, e.g., 'resnet' or ['densenet', 'resnet'].

#         Returns:
#             A decorator that registers the class.

#         Example:
#             >>> @registration.register(name="resnet18", family="resnet")
#             ... class ResNet18:
#             ...     pass
#         """

#         def decorator(cls):
#             key = (name or cls.__name__).lower()
#             fam = [family] if isinstance(family, str) else family
#             entry = {"class": cls, "family": [f.lower() for f in (fam or [])]}
#             if key in self._registry:
#                 raise KeyError(f"Backbone '{key}' is already registered")
#             self._registry[key] = entry
#             return cls

#         return decorator

#     def get_entry(self, name: str):
#         """Retrieves a backbone's registry entry by its name.

#         Args:
#             name: The name of the backbone to retrieve (case-insensitive).

#         Returns:
#             A dictionary containing the class and family of the backbone.

#         Raises:
#             KeyError: If the specified backbone is not found in the registry.
#         """
#         key = name.lower()
#         if key not in self._registry:
#             raise KeyError(f"Backbone '{name}' not found. Available: {list(self._registry.keys())}")
#         return self._registry[key]

#     def get(self, name: str):
#         """Retrieves a backbone's class by its name.

#         Args:
#             name: The name of the backbone to retrieve (case-insensitive).

#         Returns:
#             The Python class object for the specified backbone.
#         """
#         return self.get_entry(name)["class"]

#     def create(self, name, **kwargs):
#         """Instantiates a backbone model from the registry.

#         Args:
#             name: The name of the backbone to instantiate (case-insensitive).
#             **kwargs: Keyword arguments to be passed to the class constructor.

#         Returns:
#             An instance of the specified backbone class.
#         """
#         cls = self.get(name)
#         return cls(**kwargs)

#     def list(self, family=None):
#         # Initialize a rich Table object
#         table = Table(
#             title="Available Models", show_header=True, header_style="bold cyan", show_lines=True
#         )

#         # Add the columns
#         table.add_column("Models", style="dim", width=15)
#         table.add_column("Encoder Name", style="magenta")

#         # Data Aggregation Logic
#         if family is not None:
#             names = [name for name, entry in self._registry.items() if family in entry["family"]]
#             grouped = {family: names} if names else {}
#         else:
#             grouped = {}
#             for name, entry in self._registry.items():
#                 for fam in entry["family"]:
#                     grouped.setdefault(fam, []).append(name)

#         # Rich Table Population Logic
#         for fam, models in sorted(grouped.items()):
#             if not models:
#                 variants_content = "None"
#             else:
#                 # Format variants as a simple list separated by newlines
#                 # Rich will automatically render this with clean wrapping/spacing
#                 variants_content = "\n".join([f"• [green]{model}[/]" for model in models])

#             # Add the row to the rich Table
#             table.add_row(fam, variants_content)

#         # Return the wrapper object containing the rich Table
#         return TabularOutput(table)


class ModelRegistry:
    """A factory registry for backbone models.

    This class provides a centralized system for registering, retrieving, and
    instantiating backbone models, such as ResNet, DenseNet, etc. It uses a
    decorator-based approach to automatically register models when their
    respective modules are imported. This simplifies model management and
    allows for easy discovery of available backbones.
    """

    def __init__(self):
        # Registry for backbone/encoder models
        self._backbone_registry = {}

        # Registry for segmentation models
        self._segmentor_registry = {}

    def register(
        self,
        name=None,
        family=None,
        type="backbone",
        allowed_families=None,
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
            model_name = (name or cls.__name__).lower()
            if type == "backbone":
                if model_name in self._backbone_registry:
                    raise KeyError(f"Backbone '{model_name}' is already registered")

                # backbone/encoder models
                fam = [family] if isinstance(family, str) else family
                fam = [f.lower() for f in (fam or [])]
                entry = {"class": cls, "family": fam}
                self._backbone_registry[model_name] = entry

            elif type == "segmentation":
                if model_name in self._segmentor_registry:
                    raise KeyError(f"Segmentor '{model_name}' is already registered")

                # segmentation models
                families = allowed_families or getattr(cls, "ALLOWED_BACKBONE_FAMILIES", [])
                families = [f.lower() for f in families]
                entry = {"class": cls, "allowed_families": families}
                self._segmentor_registry[model_name] = entry
            else:
                raise ValueError(
                    f"Unknown model type: {type}. Must be 'backbone' or 'segmentation'."
                )

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

        if key in self._backbone_registry:
            return self._backbone_registry[key]

        if key in self._segmentor_registry:
            return self._segmentor_registry[key]

        # Debugging purpose only!
        available_backbones = list(self._backbone_registry.keys())
        available_segmentors = list(self._segmentor_registry.keys())
        all_available = sorted(available_backbones + available_segmentors)
        raise KeyError(f"Model '{name}' not found in registry. Available models: {all_available}")

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

    def _get_backbones_grouped_by_family(self):
        # Helper to group all backbone variants by their family.
        grouped = {}
        for name, entry in self._backbone_registry.items():
            for fam in entry["family"]:
                grouped.setdefault(fam, []).append(name)
        return grouped

    def _get_segmentors_grouped_by_family(self):
        # Group segmentors based on the families they support.
        grouped = {}
        for seg_name, entry in self._segmentor_registry.items():
            for fam in entry["allowed_families"]:
                grouped.setdefault(fam, []).append(seg_name)
        return grouped

    def list(self, family=None):
        # Initialize rich Table
        table = Table(
            title="Model Registry Catalog",
            show_header=True,
            header_style="bold magenta",
            show_lines=True,
        )

        # Add the columns
        table.add_column("Segmentor", style="cyan", min_width=15, justify="left")
        table.add_column("Backbone Family", style="dim", min_width=12, justify="left")
        table.add_column("Variants", style="green", justify="left")

        # Data Preparation: Group backbones and segmentors
        all_backbones = self._get_backbones_grouped_by_family()
        segmentors_by_family = self._get_segmentors_grouped_by_family()

        # Determine the complete set of families that must be processed:
        # Families used by Segmentors OR Families that have registered Backbones
        all_relevant_families = set(all_backbones.keys()) | set(segmentors_by_family.keys())
        families_to_process = sorted(list(all_relevant_families))

        if family is not None:
            family_lower = family.lower()
            families_to_process = [f for f in families_to_process if f == family_lower]

        # Rich Table Population Logic
        for fam in families_to_process:
            segmentors = sorted(segmentors_by_family.get(fam, []))

            # Process segmentors
            if segmentors:
                segmentors_content = "\n".join([f"• {seg}" for seg in segmentors])
            else:
                # Case: Backbone Family exists, but no Segmentor supports it
                segmentors_content = "[i dim]None Supported[/]"

            # Process variants (encoder_name) for this family
            models = sorted(all_backbones.get(fam, []))
            if models:
                variants_content = "\n".join([f"• {model}" for model in models])
            else:
                # Case: Segmentor supports this family, but no Backbones exist
                variants_content = "[i dim]None Available[/]"

            # Add the row for the whole family group
            table.add_row(segmentors_content, fam, variants_content)
        return TabularOutput(table)


registration = ModelRegistry()
