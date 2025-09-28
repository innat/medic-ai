import keras
from keras import ops

from medicai.utils import DescribeMixin, registration

from .vit_backbone import ViTBackbone


@keras.saving.register_keras_serializable(package="vit")
class ViTVariantsBase(keras.Model, DescribeMixin):
    """
    Vision Transformer (ViT) Model for classification.

    This class implements a Keras-based Vision Transformer (ViT) model,
    supporting both 2D and 3D inputs. It combines the ViT backbone with
    optional components like an intermediate pre-logits layer, dropout, and
    a classification head. Subclasses (like ViTVariantsBase) define the specific
    hyperparameters (patch size, num layers, hidden dim, etc.).
    """

    def __init__(
        self,
        *,
        input_shape,
        patch_size,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        use_class_token,
        include_rescaling=False,
        include_top=True,
        pooling="token",
        num_classes=1000,
        intermediate_dim=None,
        classifier_activation=None,
        dropout=0.0,
        name="vit",
        **kwargs,
    ):
        """
        Initializes the ViTVariantsBase model.

        Args:
            input_shape (tuple): Shape of the input tensor excluding batch size.
                For example, (height, width, channels) for 2D
                or (depth, height, width, channels) for 3D.
            patch_size (int or tuple): Size of the patches extracted from the input.
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads in each transformer layer.
            hidden_dim (int): Hidden dimension size of the transformer encoder (C).
            mlp_dim (int): Hidden dimension size of the MLP in transformer blocks.
            use_class_token (bool): Whether to prepend a class (CLS) token to the sequence.
            include_rescaling (bool): Whether to include a Rescaling layer at the
                start to normalize inputs. Default: False.
            include_top (bool): Whether to include the final classification layer.
                Default: True.
            pooling (str): Pooling strategy for the output if `include_top` is False.
                'token' for the CLS token (default for ViT).
                'gap' for global average pooling over sequence dimension.
            num_classes (int): Number of output classes for classification. Default: 1000.
            intermediate_dim (int, optional): Dimension of optional pre-logits dense layer.
                 If None, no intermediate layer is used.
            classifier_activation (str, optional): Activation function for the final output layer.
                Typically 'softmax' or 'sigmoid'. Default: None.
            dropout (float): Dropout rate applied before the output layer. Default: 0.0.
            name (str): Name of the model. Default: 'vit'.
            **kwargs: Additional keyword arguments passed to keras.Model.
        """
        spatial_dims = len(input_shape) - 1
        if name is None and self.__class__ is not ViTVariantsBase:
            name = f"{self.__class__.__name__}{spatial_dims}D"

        # ViT Backbone
        backbone = ViTBackbone(
            input_shape=input_shape,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            include_rescaling=include_rescaling,
            use_class_token=use_class_token,
            dropout_rate=0.0,
            attention_dropout=0.0,
            layer_norm_epsilon=1e-6,
            use_mha_bias=True,
            use_mlp_bias=True,
            use_patch_bias=True,
            name=name + "_backbone",
        )
        input = backbone.input
        x = backbone.output

        if include_top:
            # Standard ViT output is the CLS token
            x = x[:, 0]

            # Optional: intermediate (pre-logits) layer
            if intermediate_dim is not None:
                intermediate_layer = keras.layers.Dense(
                    intermediate_dim, activation="tanh", name="pre_logits"
                )
                x = intermediate_layer(x)

            # output dropout layer
            x = keras.layers.Dropout(rate=dropout, name="output_dropout")(x)

            # output layer
            output_dense = keras.layers.Dense(
                num_classes,
                activation=classifier_activation,
                dtype="float32",
                name="predictions",
            )
            x = output_dense(x)
        elif pooling == "token":
            x = x[:, 0]  # CLS token
        elif pooling == "gap":
            ndim = len(ops.shape(x))
            x = ops.mean(x, axis=list(range(1, ndim - 1)))  # mean over spatial dims

        super().__init__(inputs=input, outputs=x, name=name, **kwargs)

        # Save config
        self.pyramid_outputs = backbone.pyramid_outputs
        self.num_classes = num_classes
        self.pooling = pooling
        self.dropout = dropout
        self.use_class_token = use_class_token
        self.intermediate_dim = intermediate_dim
        self.classifier_activation = classifier_activation
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.include_rescaling = include_rescaling
        self.include_top = include_top

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape[1:],
                "num_classes": self.num_classes,
                "pooling": self.pooling,
                "use_class_token": self.use_class_token,
                "intermediate_dim": self.intermediate_dim,
                "classifier_activation": self.classifier_activation,
                "dropout": self.dropout,
                "include_rescaling": self.include_rescaling,
                "include_top": self.include_top,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="vit")
@registration.register(name="vit_base", family="vit")
class ViTBase(ViTVariantsBase):
    """
    Vision Transformer Base (ViT-B/16) model.

    ViT-B/16 is the standard base-sized model from the **Vision Transformer**
    (ViT) family. It is parameterized with the configuration specified for the
    Base variant with a 16x16 patch size in the original paper.

    This class inherits from `ViTVariantsBase` and sets the hyper-parameters
    for patch size, number of layers, heads, and hidden dimensions specific
    to the ViT-Base architecture.

    Example:
        >>> from medicai.models import ViTBase
        >>> # Load ViT-Base for ImageNet (2D input: H, W, C)
        >>> model = ViTBase(input_shape=(224, 224, 3))
        >>> # Load ViT-Base backbone (for 3D input: D, H, W, C)
        >>> backbone = ViTBase(input_shape=(32, 128, 128, 1), include_top=False)

    Reference:
        'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'
        - Paper: https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling="token",
        use_class_token=True,
        classifier_activation=None,
        name=None,
        **kwargs,
    ):
        """
        Initializes the ViT-Base model.

        This constructor automatically sets the ViT-Base-specific hyper-parameters
        and accepts configuration for the input and output head.

        Args:
            input_shape (tuple): Shape of the input tensor excluding batch size.
                This model supports 2D (H, W, C) and 3D (D, H, W, C) input.
            include_rescaling (bool): Whether to include a **Rescaling** layer at the
                start to normalize inputs.
                *Effect:* If True, applies an input pre-processing step. Default: False.
            include_top (bool): Whether to include the final **Dense classification
                layer** (the "top") on top of the feature extractor.
                *Effect:* If True, the model output is a probability distribution/logits.
                    If False, the model outputs the feature map for downstream tasks. Default: True.
            num_classes (int): Optional integer, the number of output classes. Only
                relevant if `include_top` is True.
                *Effect:* Sets the number of units in the final Dense layer. Default: 1000.
            pooling (str): Pooling strategy for the output if `include_top` is False.
               'token' for the CLS token.
                *Effect:* Collapses the sequence dimension of the feature map. Default: 'token'.
            use_class_token (bool): Whether to prepend a class (CLS) token to the sequence.
                Default: True.
            classifier_activation (str, optional): Optional activation function for the final
                Dense classification layer. Only relevant if `include_top` is True.
                *Effect:* Typically 'softmax' or 'sigmoid'. Default: None.
            name (str, optional): The name to give the Keras model.
                *Effect:* Sets the model's identifier. Default: Auto-generated as 'ViTBase{D}D'.
            **kwargs: Additional keyword arguments passed to the base `ViTVariantsBase`
                      constructor.

        Default Parameters (ViT-Base specific):
            patch_size (int): 16
                *Role:* Defines the size of the square/cube patches (e.g., 16x16 or 16x16x16) extracted
                    from the input.
                *Effect:* Determines the **sequence length** of the transformer; a smaller size results
                    in more patches and higher computational cost.
            num_layers (int): 12
                *Role:* The number of sequential **Transformer Encoder Blocks**.
                *Effect:* Defines the **depth** of the model, allowing it to capture more complex,
                    hierarchical relationships.
            num_heads (int): 12
                *Role:* The number of parallel **attention mechanisms** (heads) in each transformer
                    block's Multi-Head Attention (MHA).
                *Effect:* Allows the model to jointly attend to information from different feature
                    subspaces.
            hidden_dim (int): 768
                *Role:* The **embedding size** or **feature dimension** (C) that all tokens are projected into.
                *Effect:* Defines the **width** of the model, increasing the model's capacity and detail with
                    which token features are represented.
            mlp_dim (int): 3072
                *Role:* The hidden dimension of the intermediate **Feed-Forward Network (FFN)** within each
                    transformer block (typically 4 * hidden_dim).
                *Effect:* Controls the non-linear transformation capacity of the network.
        """
        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            use_class_token=use_class_token,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="vit")
@registration.register(name="vit_large", family="vit")
class ViTLarge(ViTVariantsBase):
    """
    Vision Transformer Large (ViT-L/16) model.

    ViT-L/16 is the large-sized model from the **Vision Transformer** (ViT) family,
    offering greater capacity and often achieving higher accuracy than ViT-Base,
    at the cost of increased computational resources.

    This class inherits from `ViTVariantsBase` and sets the hyper-parameters
    for patch size, number of layers, heads, and hidden dimensions specific
    to the ViT-Large architecture (L/16 variant).

    Example:
        >>> from medicai.models import ViTLarge
        >>> # Load ViT-Large for ImageNet (2D input: H, W, C)
        >>> model = ViTLarge(input_shape=(224, 224, 3))
        >>> # Load ViT-Large backbone (for 3D input: D, H, W, C)
        >>> backbone = ViTLarge(input_shape=(32, 128, 128, 1), include_top=False)

    Reference:
        'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'
        - Paper: https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling="token",
        use_class_token=True,
        classifier_activation=None,
        name=None,
        **kwargs,
    ):
        """
        Initializes the ViT-Large model.

        This constructor automatically sets the ViT-Large-specific hyper-parameters
        and accepts configuration for the input and output head.

        Args:
            input_shape (tuple): Shape of the input tensor excluding batch size.
                This model supports 2D (H, W, C) and 3D (D, H, W, C) input.
                *Effect:* Defines the input shape for the entire model.
            include_rescaling (bool): Whether to include a **Rescaling** layer at the
               start to normalize inputs. Default: False.
            include_top (bool): Whether to include the final **Dense classification
                layer** (the "top") on top of the feature extractor. Default: True.
            num_classes (int): Number of output classes for classification. Default: 1000.
            pooling (str): Pooling strategy for the output if `include_top` is False.
                'token' for the CLS token. Default: 'token'.
            classifier_activation (str, optional): Activation function for the final
                Dense classification layer. Default: None.
            use_class_token (bool): Whether to prepend a class (CLS) token to the sequence.
                Default: True.
            name (str, optional): The name to give the Keras model. Default: Auto-generated.
            **kwargs: Additional keyword arguments passed to the base `ViTVariantsBase` constructor.

        Default Parameters (ViT-Large specific):
            patch_size (int): 16
                *Role:* Defines the size of the square/cube patches (e.g., 16x16 or 16x16x16) extracted
                    from the input.
                *Effect:* Determines the **sequence length** of the transformer; a smaller size results
                    in more patches and higher computational cost.
            num_layers (int): 24
                *Role:* The number of sequential **Transformer Encoder Blocks**.
                *Effect:* Defines the **depth** of the model, allowing it to capture more complex,
                    hierarchical relationships.
            num_heads (int): 16
                *Role:* The number of parallel **attention mechanisms** (heads) in each transformer
                    block's Multi-Head Attention (MHA).
                *Effect:* Allows the model to jointly attend to information from different feature
                    subspaces.
            hidden_dim (int): 1024
                *Role:* The **embedding size** or **feature dimension** (C) that all tokens are projected into.
                *Effect:* Defines the **width** of the model, increasing the model's capacity and detail with
                    which token features are represented.
            mlp_dim (int): 4096
                *Role:* The hidden dimension of the intermediate **Feed-Forward Network (FFN)** within each
                    transformer block (typically 4 * hidden_dim).
                *Effect:* Controls the non-linear transformation capacity of the network.
        """
        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            use_class_token=use_class_token,
            patch_size=16,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="vit")
@registration.register(name="vit_huge", family="vit")
class ViTHuge(ViTVariantsBase):
    """
    Vision Transformer Huge (ViT-H/14) model.

    ViT-H/14 is the largest model from the original **Vision Transformer** (ViT)
    family. It boasts the highest capacity, often achieving state-of-the-art
    accuracy, but requires substantial computational resources and memory.

    This class inherits from `ViTVariantsBase` and sets the hyper-parameters
    for patch size, number of layers, heads, and hidden dimensions specific
    to the ViT-Huge architecture (H/14 variant). Note the common use of a smaller
    patch size (14) relative to the input resolution, increasing the sequence length.

    Example:
        >>> from medicai.models import ViTHuge
        >>> # Load ViT-Huge for large image classification (2D input: H, W, C)
        >>> model = ViTHuge(input_shape=(448, 448, 3))
        >>> # Load ViT-Huge backbone (for 3D input: D, H, W, C)
        >>> backbone = ViTHuge(input_shape=(64, 256, 256, 1), include_top=False)

    Reference:
        'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'
        - Paper: https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        *,
        input_shape,
        include_rescaling=False,
        include_top=True,
        num_classes=1000,
        pooling="token",
        use_class_token=True,
        classifier_activation=None,
        name=None,
        **kwargs,
    ):
        """
        Initializes the ViT-Huge model.

        This constructor automatically sets the ViT-Huge-specific hyper-parameters
        and accepts configuration for the input and output head.

        Args:
            input_shape (tuple): Shape of the input tensor excluding batch size.
                This model supports 2D (H, W, C) and 3D (D, H, W, C) input.
                *Effect:* Defines the input shape for the entire model.
            include_rescaling (bool): Whether to include a **Rescaling** layer at the
               start to normalize inputs. Default: False.
            include_top (bool): Whether to include the final **Dense classification
                layer** (the "top") on top of the feature extractor. Default: True.
            num_classes (int): Number of output classes for classification. Default: 1000.
            pooling (str): Pooling strategy for the output if `include_top` is False.
                'token' for the CLS token. Default: 'token'.
            classifier_activation (str, optional): Activation function for the final
                Dense classification layer. Default: None.
            use_class_token (bool): Whether to prepend a class (CLS) token to the sequence.
                Default: True.
            name (str, optional): The name to give the Keras model. Default: Auto-generated.
            **kwargs: Additional keyword arguments passed to the base `ViTVariantsBase` constructor.

        Default Parameters (ViT-Huge specific):
            patch_size (int): 14
                *Role:* Defines the size of the square/cube patches (e.g., 14x14 or 14x14x14) extracted
                    from the input.
                *Effect:* Determines the **sequence length** of the transformer; a smaller size results
                    in more patches and higher computational cost.
            num_layers (int): 32
                *Role:* The number of sequential **Transformer Encoder Blocks**.
                *Effect:* Defines the **depth** of the model, allowing it to capture more complex,
                    hierarchical relationships.
            num_heads (int): 16
                *Role:* The number of parallel **attention mechanisms** (heads) in each transformer
                    block's Multi-Head Attention (MHA).
                *Effect:* Allows the model to jointly attend to information from different feature
                    subspaces.
            hidden_dim (int): 1280
                *Role:* The **embedding size** or **feature dimension** (C) that all tokens are projected into.
                *Effect:* Defines the **width** of the model, increasing the model's capacity and detail with
                    which token features are represented.
            mlp_dim (int): 5120
                *Role:* The hidden dimension of the intermediate **Feed-Forward Network (FFN)** within each
                    transformer block (typically 4 * hidden_dim).
                *Effect:* Controls the non-linear transformation capacity of the network.
        """
        super().__init__(
            input_shape=input_shape,
            include_rescaling=include_rescaling,
            include_top=include_top,
            num_classes=num_classes,
            pooling=pooling,
            classifier_activation=classifier_activation,
            use_class_token=use_class_token,
            patch_size=14,
            num_layers=32,
            num_heads=16,
            hidden_dim=1280,
            mlp_dim=5120,
            name=name,
            **kwargs,
        )
