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


class ModelRegistry:
    """A unified factory registry for both Backbone (Encoder) and Segmentation models.

    This class provides a centralized system for registering, retrieving, and
    instantiating models. It uses a decorator-based approach to automatically
    register models, simplifying model management and enabling easy discovery of
    available backbones and their supported segmentation architectures.
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
        """Registers a model (backbone or segmentor) using a decorator.

        This method is designed to be used as a decorator on a class. It directs the
        model to either the backbone or segmentor registry based on the 'type' argument.

        Args:
            name: (Optional) The name to register the class under. Defaults to
                the class name (case-insensitive).
            family: (Required for type='backbone') The family of the encoder,
                e.g., 'resnet' or ['densenet'].
            type: The type of model to register. Must be 'backbone' (default) or 'segmentation'.
            allowed_families: (Required for type='segmentation') A list of backbone
                families that this segmentation model is compatible with, e.g.,
                ["resnet", "densenet"]. If None, it attempts to read the
                ALLOWED_BACKBONE_FAMILIES attribute from the decorated class.

        Returns:
            A decorator that registers the class.

        Example (Backbone):
            >>> @registration.register(name="resnet18", family="resnet")
            ... class ResNet18:
            ...     pass

        Example (Segmentor):
            >>> @registration.register(type="segmentation", name="unet")
            ... class UNet:
            ...     ALLOWED_BACKBONE_FAMILIES = ["resnet", "densenet"]
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
        """Retrieves a model's registry entry by its name (backbone or segmentor).

        Args:
            name: The name of the model to retrieve (case-insensitive).

        Returns:
            A dictionary containing the model's class and metadata
            (either 'family' for backbones or 'allowed_families' for segmentors).

        Raises:
            KeyError: If the specified model is not found in either registry.
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
        """Retrieves a registered model's class by its name.

        Args:
            name: The name of the model to retrieve (case-insensitive).

        Returns:
            The Python class object for the specified model.
        """
        return self.get_entry(name)["class"]

    def create(self, name, **kwargs):
        """Instantiates a model from the registry.

        This method can instantiate either a backbone or a segmentation model.

        Args:
            name: The name of the model to instantiate (case-insensitive).
            **kwargs: Keyword arguments to be passed to the class constructor.

        Returns:
            An instance of the specified model class.
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
        """Prints a rich Table catalog of all registered models.

        Models are grouped by their backbone family, showing which segmentors
        support that family and what backbone variants are available.

        Args:
            family: (Optional) Filter the output to show only a specific backbone family.

        Returns:
            A TabularOutput object containing the formatted rich Table.
        """
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
