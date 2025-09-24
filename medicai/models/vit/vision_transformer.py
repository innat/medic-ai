import keras
from keras import ops

from medicai.utils import registration

from .vit_backbone import ViTBackbone


@keras.saving.register_keras_serializable(package="vit")
@registration.register(family="vit")
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
        """
        Vision Transformer (ViT) model for classification.

        This class implements a Keras-based Vision Transformer (ViT) model,
        supporting both 2D and 3D inputs. The model consists of a ViT backbone,
        optional intermediate pre-logits layer, dropout, and a classification head.

        Args:
            input_shape (tuple): Shape of the input tensor excluding batch size.
                                For example, (height, width, channels) for 2D
                                or (depth, height, width, channels) for 3D.
            num_classes (int): Number of output classes for classification.
            patch_size (int or tuple): Size of the patches extracted from the input.
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads in each transformer layer.
            hidden_dim (int): Hidden dimension size of the transformer encoder.
            mlp_dim (int): Hidden dimension size of the MLP in transformer blocks.
            pooling (str): Pooling strategy for the output. "token" for CLS token,
                        "gap" for global average pooling over spatial dimensions.
            intermediate_dim (int, optional): Dimension of optional pre-logits dense layer.
                                            If None, no intermediate layer is used.
            classifier_activation (str, optional): Activation function for the output layer.
            dropout (float): Dropout rate applied before the output layer.
            name (str): Name of the model.
            **kwargs: Additional keyword arguments passed to keras.Model.

        Example:
            # 2D ViT for 10-class classification
            model = ViT(input_shape=(224, 224, 3), num_classes=10, patch_size=16)

            # 3D ViT with intermediate layer and global average pooling
            model = ViT(
                input_shape=(16, 128, 128, 1),
                num_classes=5,
                patch_size=4,
                intermediate_dim=512,
                pooling="gap",
            )
        """

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
