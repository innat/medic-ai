import keras
from keras import layers

from medicai.utils import DescribeMixin

from .vit_layers import ViTEncoderBlock, ViTPatchingAndEmbedding


class ViTBackbone(keras.Model, DescribeMixin):
    """
    The backbone is constructed in the following stages:

    1. An input layer is created from ``input_shape``.
    2. An optional ``Rescaling`` layer normalizes raw image intensities.
    3. The input is split into non-overlapping patches and projected into the
       transformer hidden dimension. An optional class token can be prepended
       to the token sequence.
    4. Dropout is applied to the patch embeddings.
    5. A stack of transformer encoder blocks is applied to the token sequence.
       After each stage, the current token sequence is stored in
       ``pyramid_outputs``.
    6. A final layer normalization is applied to produce the backbone output.

    Args:
        input_shape: A tuple specifying the input shape of the model, not
            including the batch size. This can describe either 2D or 3D
            inputs.
        patch_size: An integer or tuple specifying the patch size used to
            split the input before embedding.
        num_layers: An integer specifying the number of transformer
            encoder blocks.
        num_heads: An integer specifying the number of attention heads in
            each transformer block.
        hidden_dim: An integer specifying the hidden dimension of the
            token embeddings.
        mlp_dim: An integer specifying the hidden dimension of the MLP
            sublayers inside each transformer block.
        use_class_token: A boolean indicating whether to prepend a class
            token to the token sequence.
        include_rescaling: A boolean indicating whether to include a
            ``Rescaling`` layer at the beginning of the model.
        dropout_rate: A float specifying the dropout rate applied after
            patch embedding and inside the transformer blocks.
        attention_dropout: A float specifying the attention dropout rate.
        layer_norm_epsilon: A float specifying the epsilon value used in
            layer normalization.
        use_mha_bias: A boolean indicating whether to use bias terms in
            multi-head attention projections.
        use_mlp_bias: A boolean indicating whether to use bias terms in
            the MLP layers.
        use_patch_bias: A boolean indicating whether to use bias terms in
            the patch embedding projection.
        name: (Optional) The name of the model.

    Returns:
        A ``keras.Model`` whose forward pass returns the final backbone
        feature tensor. Intermediate multi-scale features are available in
        the ``pyramid_outputs`` attribute.

    Examples:
        .. code-block:: python

            import torch
            from medicai.models.vit import ViTBackbone

            model = ViTBackbone(
                input_shape=(224, 224, 3),
                patch_size=16,
                num_layers=4,
                num_heads=8,
                hidden_dim=256,
                mlp_dim=512,
                use_class_token=False
            )
            x = torch.randn((1, 224, 224, 3))
            y = model(x)
            print(y.shape) # torch.Size([1, 196, 256])


    References:
        - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.
          `arXiv:2010.11929 <https://arxiv.org/abs/2010.11929>`_

    """

    def __init__(
        self,
        input_shape,
        patch_size,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        use_class_token,
        include_rescaling=False,
        dropout_rate=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-6,
        use_mha_bias=True,
        use_mlp_bias=True,
        use_patch_bias=True,
        name=None,
        **kwargs,
    ):
        """
        Initializes the ViT backbone model.

        Args:
            input_shape (tuple): Shape of the input tensor excluding batch size.
                For example, (height, width, channels) for 2D
                or (depth, height, width, channels) for 3D.
            patch_size (int or tuple): Size of the patches extracted from the input.
                If int, the same size is used for all spatial dims.
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads per transformer layer.
            hidden_dim (int): Hidden dimension size of the transformer encoder (C).
            mlp_dim (int): Hidden dimension size of the MLP in each transformer block.
            include_rescaling (bool): Whether to include a Rescaling layer at the
               start to normalize inputs (1/255). Default: False.
            dropout_rate (float): Dropout rate applied after patch embedding and in MLPs.
            attention_dropout (float): Dropout rate for the attention weights.
            layer_norm_epsilon (float): Epsilon for layer normalization.
            use_mha_bias (bool): Whether to use bias in multi-head attention.
            use_mlp_bias (bool): Whether to use bias in the MLP layers.
            use_class_token (bool): Whether to prepend a class (CLS) token to the sequence.
            use_patch_bias (bool): Whether to use bias in the patch embedding layer.
            name (str, optional): Name of the model.
            **kwargs: Additional keyword arguments passed to keras.Model.

        Example:
            # 2D ViT backbone (ViT-Base params)
            from medicai.models import ViTBackbone
            backbone = ViTBackbone(
                input_shape=(224, 224, 3),
                patch_size=16,
                num_layers=12,
                num_heads=12,
                hidden_dim=768,
                mlp_dim=3072,
            )

            # 3D ViT backbone
            from medicai.models import ViTBackbone
            backbone3d = ViTBackbone(
                input_shape=(16, 128, 128, 1),
                patch_size=(4, 16, 16),
                num_layers=8,
                num_heads=8,
                hidden_dim=512,
                mlp_dim=2048,
            )
        """

        # === Spatial dims detection ===
        spatial_dims = len(input_shape) - 1  # 2 or 3
        *image_shape, num_channels = input_shape

        if None in image_shape:
            raise ValueError(
                f"Image shape must have defined spatial dimensions. " f"Found {input_shape}"
            )

        # Normalize patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * spatial_dims
        if len(patch_size) != spatial_dims:
            raise ValueError(
                f"patch_size length {len(patch_size)} does not match "
                f"image spatial dims {spatial_dims}."
            )

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        # Validate divisibility
        for im_dim, p_dim in zip(input_shape, patch_size):
            if im_dim % p_dim != 0:
                raise ValueError(f"Image dimension {im_dim} not divisible by patch size {p_dim}.")

        # === Functional Model ===
        inputs = keras.Input(shape=input_shape, name="images")
        pyramid_outputs = {}

        x = inputs
        if include_rescaling:
            x = layers.Rescaling(1.0 / 255)(x)

        x = ViTPatchingAndEmbedding(
            image_size=image_shape,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_channels=num_channels,
            use_class_token=use_class_token,
            use_patch_bias=use_patch_bias,
            name="vit_patching_and_embedding",
        )(x)
        pyramid_outputs["P1"] = x

        x = keras.layers.Dropout(dropout_rate, name="vit_dropout")(x)

        for i in range(num_layers):
            encoder_block = ViTEncoderBlock(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                mlp_dim=mlp_dim,
                dropout_rate=dropout_rate,
                use_mha_bias=use_mha_bias,
                use_mlp_bias=use_mlp_bias,
                attention_dropout=attention_dropout,
                layer_norm_epsilon=layer_norm_epsilon,
                name=f"vit_feature{i + 1}",
            )
            x = encoder_block(x)
            pyramid_outputs[f"P{i+2}"] = x

        output = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            name="vit_ln",
        )(x)

        super().__init__(inputs=inputs, outputs=output, name=name or "ViTBackbone", **kwargs)

        # === Config ===
        self.pyramid_outputs = pyramid_outputs
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.include_rescaling = include_rescaling
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_mha_bias = use_mha_bias
        self.use_mlp_bias = use_mlp_bias
        self.use_class_token = use_class_token
        self.use_patch_bias = use_patch_bias

    def get_config(self):
        return {
            "input_shape": self.input_shape[1:],
            "patch_size": self.patch_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "mlp_dim": self.mlp_dim,
            "include_rescaling": self.include_rescaling,
            "dropout_rate": self.dropout_rate,
            "attention_dropout": self.attention_dropout,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "use_mha_bias": self.use_mha_bias,
            "use_mlp_bias": self.use_mlp_bias,
            "use_class_token": self.use_class_token,
            "use_patch_bias": self.use_patch_bias,
        }
