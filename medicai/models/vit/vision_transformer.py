import keras
from keras import ops

from .vit_backbone import ViTBackbone


class ViT(keras.Model):
    def __init__(
        self,
        *,
        input_shape,
        num_classes,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        pooling="token",
        intermediate_dim=None,
        classifier_activation=None,
        dropout=0.0,
        name="vit",
        **kwargs,
    ):

        # === Backbone ===
        self.backbone = ViTBackbone(
            input_shape=input_shape,  # (h, w, c) or (d, h, w, c)
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout_rate=0.0,
            attention_dropout=0.0,
            layer_norm_epsilon=1e-6,
            use_mha_bias=True,
            use_mlp_bias=True,
            use_class_token=True,
            use_patch_bias=True,
            name=name + "_backbone",
            dtype=None,
        )

        # === Optional intermediate (pre-logits) layer ===
        self.intermediate_layer = None
        if intermediate_dim is not None:
            self.intermediate_layer = keras.layers.Dense(
                intermediate_dim, activation="tanh", name="pre_logits"
            )

        self.dropout = keras.layers.Dropout(
            rate=dropout,
            name="output_dropout",
        )
        self.output_dense = keras.layers.Dense(
            num_classes,
            activation=classifier_activation,
            dtype="float32",
            name="predictions",
        )

        # === Functional model wiring ===
        inputs = keras.Input(shape=input_shape, name="inputs")
        x = self.backbone(inputs)

        # Pooling
        if pooling == "token":
            x = x[:, 0]  # CLS token
        elif pooling == "gap":
            ndim = len(ops.shape(x))
            x = ops.mean(x, axis=list(range(1, ndim - 1)))  # mean over spatial dims
        else:
            raise ValueError(f"Invalid pooling type: {pooling}")

        if self.intermediate_layer is not None:
            x = self.intermediate_layer(x)

        x = self.dropout(x)
        outputs = self.output_dense(x)

        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        # Save config
        self.num_classes = num_classes
        self.pooling = pooling
        self.intermediate_dim = intermediate_dim
        self.classifier_activation = classifier_activation
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim

    def get_config(self):
        return {
            "input_shape": self.input_shape[1:],
            "num_classes": self.num_classes,
            "pooling": self.pooling,
            "intermediate_dim": self.intermediate_dim,
            "classifier_activation": self.classifier_activation,
            "patch_size": self.patch_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "mlp_dim": self.mlp_dim,
        }
