import keras

from medicai.layers import ViTEncoderBlock, ViTPatchingAndEmbedding


class ViTBackbone(keras.Model):
    def __init__(
        self,
        input_shape,  # (h, w, c) for 2D or (d, h, w, c) for 3D
        patch_size,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout_rate=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-6,
        use_mha_bias=True,
        use_mlp_bias=True,
        use_class_token=True,
        use_patch_bias=True,
        name=None,
        **kwargs,
    ):
        # === Spatial dims detection ===
        spatial_dims = len(input_shape) - 1  # 2 or 3
        image_shape = input_shape[:-1]  # (h, w) or (d, h, w)
        num_channels = input_shape[-1]

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

        # Validate divisibility
        for im_dim, p_dim in zip(input_shape, patch_size):
            if im_dim % p_dim != 0:
                raise ValueError(f"Image dimension {im_dim} not divisible by patch size {p_dim}.")

        # === Functional Model ===
        inputs = keras.layers.Input(shape=input_shape, name="images")

        x = ViTPatchingAndEmbedding(
            image_size=image_shape,  # (h, w) or (d, h, w)
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_channels=num_channels,
            use_class_token=use_class_token,
            use_patch_bias=use_patch_bias,
            name="vit_patching_and_embedding",
        )(inputs)

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

        output = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            name="vit_ln",
        )(x)

        super().__init__(inputs=inputs, outputs=output, name=name or "ViTBackbone", **kwargs)

        # === Config ===
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
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
            "dropout_rate": self.dropout_rate,
            "attention_dropout": self.attention_dropout,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "use_mha_bias": self.use_mha_bias,
            "use_mlp_bias": self.use_mlp_bias,
            "use_class_token": self.use_class_token,
            "use_patch_bias": self.use_patch_bias,
        }
