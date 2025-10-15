import inspect


class DescriptionObject:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


class DescribeMixin:
    _skip_keys = {"layers", "input_layers", "output_layers", "weights", "layer_names"}

    @classmethod
    def class_describe(cls, pretty: bool = True):
        """Return class-level description including docstring and __init__ args."""
        base_doc = inspect.cleandoc(cls.__doc__ or "No description available.")
        name = cls.__name__

        if pretty:
            lines = [f"📌 Class: {name}", "\n📝 Description:", base_doc]

            # Optional: show allowed backbones if defined
            if hasattr(cls, "ALLOWED_BACKBONE_FAMILIES"):
                lines.append("\n🧩 Allowed Backbone Families:")
                for fam in cls.ALLOWED_BACKBONE_FAMILIES:
                    lines.append(f"  • {fam}")

            # Constructor arguments
            lines.append("\n⚙️ Constructor Arguments:")
            try:
                sig = inspect.signature(cls.__init__)
                for pname, param in sig.parameters.items():
                    if pname == "self":
                        continue
                    # default value
                    default = (
                        f"= {param.default!r}"
                        if param.default is not inspect.Parameter.empty
                        else ""
                    )
                    # annotation
                    if param.annotation != inspect.Parameter.empty:
                        annot_name = getattr(param.annotation, "__name__", repr(param.annotation))
                        annot = f": {annot_name}"
                    else:
                        annot = ""
                    lines.append(f"  {pname}{annot} {default}".rstrip())

                init_doc = inspect.cleandoc(cls.__init__.__doc__ or "")
                if init_doc:
                    lines.append(f"\n📘 Details Constructor Arguments:\n{init_doc}")
            except (ValueError, TypeError):
                lines.append("  <unable to inspect constructor>")

            return DescriptionObject("\n".join(lines))

        # Machine-friendly dict version
        desc = {"name": name, "doc": base_doc}
        if hasattr(cls, "ALLOWED_BACKBONE_FAMILIES"):
            desc["allowed_backbones"] = cls.ALLOWED_BACKBONE_FAMILIES
        try:
            desc["args"] = {
                pname: str(param)
                for pname, param in inspect.signature(cls.__init__).parameters.items()
                if pname != "self"
            }
        except (ValueError, TypeError):
            desc["args"] = {}
        return desc

    def _format_encoder_details(self, encoder_desc, indent="  "):
        """Helper to format the detailed configuration of an instantiated encoder."""
        irrelevant_keys = {"num_classes", "classifier_activation", "pooling", "include_top"}

        # Start the description with the encoder's class name and an opening parenthesis
        # Note: We use the next level of indentation (indent + "  ") for the parameters
        lines = [f"{encoder_desc['class']}("]

        # Add the parameters, skipping irrelevant ones
        for nk, nv in encoder_desc["params"].items():
            if nk in irrelevant_keys:
                continue
            # Use !r for repr and double indentation for parameters
            lines.append(f"{indent}  • {nk}: {nv!r}")

        # Add the closing parenthesis
        lines.append(")")
        return lines

    def _format_param(self, k, v, encoder_desc=None, indent="  "):
        # Special case: encoder key
        # Check if we have encoder details and the current key is one of the encoder keys
        if encoder_desc is not None and k in {"encoder", "encoder_name"}:

            # Get the formatted lines for the nested encoder config
            # Pass the current indent to the helper for correct parameter alignment
            encoder_lines = self._format_encoder_details(encoder_desc, indent)
            # Start the main block with the 'encoder:' label and the first line of the details
            lines = [f"{indent}• encoder: {encoder_lines[0]}"]
            # Add all the detailed parameter lines
            lines.extend(encoder_lines[1:-1])
            # Add the closing parenthesis line
            lines.append(f"{indent}  {encoder_lines[-1]}")

            if k == "encoder_name":
                # If it was passed by name, append the original 'encoder_name' parameter line at the end
                lines.append(f"{indent}• encoder_name: {v!r}")

            return "\n".join(lines)

        if isinstance(v, (dict, list, tuple)):
            return f"{indent}• {k}: {v}"
        else:
            return f"{indent}• {k}: {v}"

    def instance_describe(self, pretty: bool = True):
        name = self.__class__.__name__

        # if encoder attribute exists and has instance_describe
        encoder = getattr(self, "encoder", None)
        encoder_desc = None

        params = {}
        if hasattr(self, "get_config") and callable(self.get_config):
            params = self.get_config()

        # remove unwanted internal keys
        params = {k: v for k, v in params.items() if k not in self._skip_keys}

        if encoder is not None and hasattr(encoder, "instance_describe"):
            encoder_desc = encoder.instance_describe(pretty=False)
        elif "encoder_name" in params:
            if getattr(self, "encoder", None) and hasattr(self.encoder, "instance_describe"):
                encoder_desc = self.encoder.instance_describe(pretty=False)

        if not pretty:
            return {"class": name, "params": params}

        # pretty string
        lines = [f"Instance of {name}"]
        if not params:
            lines.append("  • <no config available>")
        else:
            for k, v in params.items():
                lines.append(self._format_param(k, v, encoder_desc=encoder_desc))
        return DescriptionObject("\n".join(lines))
