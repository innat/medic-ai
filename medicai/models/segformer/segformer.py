import keras
from keras import layers, ops

from medicai.layers import ConvBnAct, ResizingND
from medicai.utils import (
    DescribeMixin,
    get_conv_layer,
    get_norm_layer,
    registration,
    resolve_encoder,
    validate_activation,
)


@keras.saving.register_keras_serializable(package="segformer")
@registration.register(name="segformer", type="segmentation")
class SegFormer(keras.Model, DescribeMixin):
    """
    SegFormer can be constructed either from a registered encoder name or
    from a pre-built encoder instance. The encoder must expose a
    ``pyramid_outputs`` dictionary containing four feature levels:
    ``P1``, ``P2``, ``P3``, and ``P4``.

    The decoder uses all four encoder feature maps:

    1. Each feature level is projected to a shared embedding dimension.
    2. The projected features are resized to the spatial resolution of ``P1``.
    3. The resized features are concatenated and fused with a convolution
       block before producing segmentation logits.
    4. The final prediction is resized back to the input spatial resolution.

    Args:
        encoder: Optional pre-built Keras model to use as the encoder. It must
            expose ``pyramid_outputs`` with the required four feature levels.
        encoder_name: Optional name of a registered MiT backbone to build and
            use as the encoder.
        input_shape: Optional input shape excluding the batch dimension.
            Required when ``encoder_name`` is used. This can describe either
            2D or 3D inputs.
        num_classes: Number of segmentation classes. Must be greater than
            zero.
        classifier_activation: Activation function used by the final
            segmentation head.
        decoder_head_embedding_dim: Embedding dimension used to project each
            encoder feature map inside the decoder head.
        dropout: Dropout rate applied in the decoder fusion block.
        name: Optional model name.
        **kwargs: Additional keyword arguments passed to ``keras.Model``.

    Examples:
        .. code-block:: python

            import jax
            import jax.numpy as jnp
            from medicai.models import SegFormer

            model = SegFormer(
                encoder_name="mit_b0",
                input_shape=(224, 224, 3),
                num_classes=2,
                classifier_activation="softmax",
            )

            key = jax.random.PRNGKey(0) 
            x = jax.random.normal(key, (1, 224, 224, 3))
            y = model(x)
            print(y.shape) # (1, 224, 224, 2)

    .. rubric:: Encoder stages
       :class: api-subheading

    **SegFormer** always uses all ``4`` encoder stages according to the official literature. There is no
    ``encoder_depth`` argument in this implementation.

    .. rubric:: Custom encoder
       :class: api-subheading

    When providing a custom encoder through ``encoder``, ensure that:

    1. It defines a ``pyramid_outputs`` dictionary with ``P1`` through ``P4``.
    2. These features follow the expected multi-scale hierarchy.
    3. The input spatial dimensions are equal across all axes, since this
       implementation currently expects **square** 2D inputs or **cubic** 3D inputs.

    Example:
        Build the model from a custom encoder with convnext which already gives ``4`` encoder stages::

            import jax
            import jax.numpy as jnp
            from medicai.models import SegFormer

            backbone = ConvNeXtV2Large(
                input_shape=(96, 96, 96, 3),
                include_top=False,
            )

            model = SegFormer(
                encoder=backbone,
                num_classes=3,
            )

            key = jax.random.PRNGKey(0) 
            x = jax.random.normal(key, (1, 96, 96, 96, 3))
            y = model(x)
            print(y.shape) # (1, 96, 96, 96, 3)

        Build the model from a custom resnet encoder with ``5`` encoder
        stages. In this case, **SegFormer** uses only ``P1`` through ``P4``::

            import jax
            import jax.numpy as jnp
            from medicai.models import ResNetBackbone, SegFormer

            backbone = ResNetBackbone(
                input_shape=(224, 224, 3),
                conv_filters=[32],
                conv_kernel_sizes=[7],
                num_filters=[64, 128, 256, 512],
                num_blocks=[3, 4, 6, 3],
                num_strides=[1, 2, 2, 2],
                block_type="bottleneck_block",
            )

            model = SegFormer(
                encoder=backbone,
                num_classes=5,
            )

            key = jax.random.PRNGKey(0) 
            x = jax.random.normal(key, (1, 224, 224, 3))
            y = model(x)
            print(y.shape) # (1, 224, 224, 5)

    Returns:
        A ``keras.Model`` whose forward pass returns a segmentation tensor of
        shape ``(batch_size, ..., num_classes)`` at the input spatial
        resolution.

    References:
        - SegFormer: Simple and Efficient Design for Semantic Segmentation with
          Transformers. NeurIPS 2021. `arXiv:2105.15203 <https://arxiv.org/abs/2105.15203>`_
        - SegFormer3D: an Efficient Transformer for 3D Medical Image
          Segmentation. `arXiv:2404.10156 <https://arxiv.org/abs/2404.10156>`_
    """

    ALLOWED_BACKBONE_FAMILIES = ["mit"]

    def __init__(
        self,
        *,
        input_shape=None,
        encoder_name=None,
        encoder=None,
        num_classes=1,
        classifier_activation=None,
        decoder_head_embedding_dim=256,
        dropout=0.0,
        name=None,
        **kwargs,
    ):
        """
        Initializes the SegFormer model.

        The encoder can be provided either as an instantiated Keras model (`encoder`)
        or by its registered name (`encoder_name`), in which case `input_shape` must be provided.

        Args:
            input_shape (tuple, optional): The shape of the input data, excluding the batch dimension.
                Required if `encoder_name` is provided. Format is (H, W, C) for 2D or (D, H, W, C) for 3D.
            encoder: (Optional) A Keras model to use as the encoder (backbone).
                This argument is intended for passing a custom or pre-trained
                model. If provided, the model must have a `pyramid_outputs` attribute,
                which should be a dictionary of intermediate feature vectors from shallow
                to deep layers (e.g., `'P1'`, `'P2'`, ...).
            encoder_name: (Optional) A string specifying the name of a
                pre-configured backbone from the `medicai.models.list_models()` to use as
                the encoder. This is a convenient option for using a backbone from
                the library without having to instantiate it manually.
            num_classes (int, optional): The number of output classes for segmentation. Default: 1.
            classifier_activation (str, optional): The activation function for the final output layer.
                Typically 'softmax' for multi-class or 'sigmoid' for multi-label/binary segmentation.
                Default: None.
            decoder_head_embedding_dim (int, optional): The hidden dimension used for linear embedding
                of the feature maps in the decoder head before fusion. Controls the capacity of the
                lightweight MLP decoder. Default: 256.
            dropout (float, optional): Dropout rate applied after the fusion convolution in the decoder head.
                Regularizes the decoder to prevent overfitting. Default: 0.0.
            name (str, optional): The name of the Keras model.
                Sets the model's identifier. Default: Auto-generated as "SegFormer{D}D".
            **kwargs: Standard Keras Model keyword arguments.
        """
        encoder, input_shape = resolve_encoder(
            encoder=encoder,
            encoder_name=encoder_name,
            input_shape=input_shape,
            allowed_families=SegFormer.ALLOWED_BACKBONE_FAMILIES,
        )
        # verify input activation.
        classifier_activation = validate_activation(classifier_activation)

        if not (0 <= dropout <= 1):
            raise ValueError("dropout should be between 0 and 1.")

        # number of classes must be positive.
        if num_classes <= 0:
            raise ValueError(
                f"Number of classes (`num_classes`) must be greater than 0, "
                f"but received {num_classes}."
            )

        inputs = encoder.input
        spatial_dims = len(input_shape) - 1

        # Check that the spatial dimensions are all equal.
        spatial_shapes = list(input_shape[:spatial_dims])
        if not all(x == spatial_shapes[0] for x in spatial_shapes):
            raise ValueError(
                f"Input shape {input_shape} is not square or cubic. "
                "SegFormer currently only supports inputs with equal spatial dimensions "
                "for proper hierarchical downsampling and reshaping."
            )

        # Get intermediate vectores
        pyramid_outputs = encoder.pyramid_outputs

        # SegFormer needs 4 skip connection layers
        required_keys = {"P1", "P2", "P3", "P4"}
        if not required_keys.issubset(pyramid_outputs.keys()):
            raise ValueError(
                f"The encoder's `pyramid_outputs` is missing one or more required keys. "
                f"Required: {required_keys}, Available: {set(pyramid_outputs.keys())}"
            )

        skips = [pyramid_outputs.get(f"P{i+1}") for i in range(4)]
        skips = skips + [inputs]

        # build_decoder method
        decoder_head = self.build_decoder(
            num_classes, decoder_head_embedding_dim, spatial_dims, dropout
        )
        outputs = decoder_head(skips)
        outputs = layers.Activation(
            classifier_activation,
            dtype="float32",
            name="predictions",
        )(outputs)

        super().__init__(
            inputs=inputs, outputs=outputs, name=name or f"SegFormer{spatial_dims}D", **kwargs
        )

        self._input_shape = input_shape
        self.num_classes = num_classes
        self.encoder_name = encoder_name
        self.encoder = encoder
        self.dropout = dropout
        self.decoder_head_embedding_dim = decoder_head_embedding_dim
        self.classifier_activation = classifier_activation

    def build_decoder(self, num_classes, decoder_head_embedding_dim, spatial_dims, dropout):
        """
        Constructs the lightweight MLP decoder head as a callable function.

        This decoder head performs four main steps:
        1. Linear Embedding: Each of the four multi-scale feature maps (P1-P4) is
           processed by a 1x1 convolution (implemented as Dense layer after flattening)
           to unify the channel dimension to `decoder_head_embedding_dim`.
        2. Upsampling: Feature maps from P2, P3, and P4 are upsampled to the resolution
           of the highest-resolution feature map (P1).
        3. Feature Fusion: All four feature maps are concatenated and passed through
           a single 3x3 (or 3D equivalent) fusion convolution block.
        4. Final Prediction: A final 1x1 convolution is used to predict the class scores,
           followed by upsampling to the original input resolution.

        Args:
            num_classes (int): The number of output channels for the final prediction.
            decoder_head_embedding_dim (int): The hidden dimension for the MLP/linear
                embedding layers.
            spatial_dims (int): 2 for 2D or 3 for 3D inputs.
            dropout (float): Dropout rate to apply in the decoder fusion block.

        Returns:
            function: A Keras-style function that takes the list of skip connections
                      and returns the final segmentation output.
        """

        def apply(inputs):
            c1, c2, c3, c4, original_input = inputs

            # Get target spatial shape from c1
            target_spatial_shape = ops.shape(c1)[1:-1]

            # Process each feature level with linear embedding and resize to c1 size
            # stage 1
            c4_shape = ops.shape(c4)
            c4 = self.linear_embedding(c4, decoder_head_embedding_dim)
            c4 = self.reshape_to_spatial(c4, c4_shape)
            c4 = self.resize_to_target(c4, target_spatial_shape, spatial_dims)

            # stage 2
            c3_shape = ops.shape(c3)
            c3 = self.linear_embedding(c3, decoder_head_embedding_dim)
            c3 = self.reshape_to_spatial(c3, c3_shape)
            c3 = self.resize_to_target(c3, target_spatial_shape, spatial_dims)

            # stage 3
            c2_shape = ops.shape(c2)
            c2 = self.linear_embedding(c2, decoder_head_embedding_dim)
            c2 = self.reshape_to_spatial(c2, c2_shape)
            c2 = self.resize_to_target(c2, target_spatial_shape, spatial_dims)

            # stage 4
            c1_shape = ops.shape(c1)
            c1 = self.linear_embedding(c1, decoder_head_embedding_dim)
            c1 = self.reshape_to_spatial(c1, c1_shape)

            # Fuse all features (channel-last: concatenate along last axis)
            x = layers.Concatenate(axis=-1)([c1, c2, c3, c4])

            # Fusion convolution
            x = ConvBnAct(
                filters=decoder_head_embedding_dim,
                kernel_size=1,
                strides=1,
                padding="same",
                normalization="batch",
                activation="relu",
                name="linear_fuse_conv",
            )(x)
            x = layers.Dropout(dropout)(x)

            # Final prediction
            x = get_conv_layer(
                spatial_dims=spatial_dims, layer_type="conv", filters=num_classes, kernel_size=1
            )(x)

            # Get output spatial shape from original input
            output_spatial_shape = ops.shape(original_input)[1:-1]
            x = self.resize_to_target(x, output_spatial_shape, spatial_dims)
            return x

        return apply

    def linear_embedding(self, x, hidden_dims):
        spatial_shape_tensor = ops.shape(x)[1:-1]
        num_patches = int(ops.prod(spatial_shape_tensor))
        x = layers.Reshape((num_patches, ops.shape(x)[-1]))(x)
        x = layers.Dense(hidden_dims)(x)
        x = get_norm_layer("layer", epsilon=1e-5)(x)
        return x

    def reshape_to_spatial(self, x, target_shape):
        spatial_shape = target_shape[1:-1]
        x = ops.reshape(x, [-1, *spatial_shape, ops.shape(x)[-1]])
        return x

    def resize_to_target(self, x, target_spatial_shape, spatial_dims):
        uid = keras.backend.get_uid(prefix="resize_op")
        x = ResizingND(
            target_shape=target_spatial_shape,
            interpolation="bilinear" if spatial_dims == 2 else "trilinear",
            name=f"resize_op{uid}",
        )(x)
        return x

    def get_config(self):
        config = {
            "input_shape": self._input_shape,
            "encoder_name": self.encoder_name,
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "decoder_head_embedding_dim": self.decoder_head_embedding_dim,
            "dropout": self.dropout,
        }

        if self.encoder_name is None and self.encoder is not None:
            config.update({"encoder": keras.saving.serialize_keras_object(self.encoder)})
        return config

    @classmethod
    def from_config(cls, config):
        if "encoder" in config and isinstance(config["encoder"], dict):
            config["encoder"] = keras.layers.deserialize(config["encoder"])
        return super().from_config(config)
