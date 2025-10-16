import keras
from keras import ops

from medicai.blocks import UnetOutBlock, UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from medicai.utils import DescribeMixin, registration, resolve_encoder


@keras.saving.register_keras_serializable(package="unetr")
@registration.register(name="unetr", type="segmentation")
class UNETR(keras.Model, DescribeMixin):
    """
    UNETR: U-Net with a Vision Transformer (ViT) backbone for 3D/2D medical image segmentation.

    UNETR integrates a ViT encoder as the backbone with a UNet-style decoder, using
    projection upsampling blocks and skip connections from intermediate transformer layers.
    It is designed to leverage the global context-modeling power of Transformers for
    high-resolution tasks like medical image segmentation.

    The model supports both 2D (H, W, C) and 3D (D, H, W, C) inputs, depending on the
    dimensional configuration of the injected Vision Transformer encoder.

    Example:
    >>> import tensorflow as tf # Assuming Keras backend uses TensorFlow for this example
    >>> from your_module import UNETR
    >>> # 3D UNETR for 3-class segmentation
    >>> model_3d = UNETR(
    ...     input_shape=(16, 128, 128, 1),
    ...     encoder_name="vit_base", # Automatically resolves and builds the ViT-Base encoder
    ...     num_classes=3,
    ...     feature_size=16,
    ...     norm_name="instance",
    ... )
    >>> output_3d = model_3d(tf.random.normal((1, 16, 128, 128, 1)))
    >>> print(output_3d.shape)
    (1, 16, 128, 128, 3) # Example output shape for 3D

    >>> # 2D UNETR for binary segmentation (e.g., cell/background)
    >>> model_2d = UNETR(
    ...     input_shape=(256, 256, 3),
    ...     encoder_name="vit_large",
    ...     num_classes=1,
    ...     classifier_activation="sigmoid",
    ...     feature_size=32,
    ...     norm_name="batch",
    ... )
    >>> output_2d = model_2d(tf.random.normal((1, 256, 256, 3)))
    >>> print(output_2d.shape)
    (1, 256, 256, 1) # Example output shape for 2D

    Reference:
        'UNETR: Transformers for 3D Medical Image Segmentation'
        - Paper: https://arxiv.org/abs/2103.10504
    """

    ALLOWED_BACKBONE_FAMILIES = ["vit"]

    def __init__(
        self,
        *,
        input_shape=None,
        encoder_name=None,
        encoder=None,
        num_classes=1,
        classifier_activation=None,
        feature_size=16,
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.0,
        name="UNETR",
        **kwargs,
    ):
        """
        Initializes the UNETR model by setting up the Vision Transformer encoder and the UNet decoder.

        Args:
            input_shape (tuple, optional): Shape of the input tensor excluding batch size.
                For example, (height, width, channels) for 2D
                or (depth, height, width, channels) for 3D.
                *Required if `encoder` is None.*
            encoder_name (str, optional): The name of a pre-registered ViT encoder variant
                (e.g., 'vit_base', 'vit_large', 'vit_huge') to use as the backbone.
                *Used if `encoder` is None.*
            encoder (keras.Model, optional): An already initialized ViT backbone model.
                If provided, `input_shape` and `encoder_name` are ignored.
            num_classes (int): Number of output segmentation classes.
                *Effect:* Sets the channel depth of the final output layer. Default: 1.
            classifier_activation (str, optional): Activation function applied to the output layer.
                *Effect:* Typically 'sigmoid' for binary/multi-label or 'softmax' for multi-class
                segmentation. Default: None.
            feature_size (int): Base number of feature channels in decoder blocks. The channels
                will be scaled up (e.g., `feature_size`, `2*feature_size`, etc.). Default: 16.
            norm_name (str): Type of normalization for decoder blocks ("instance", "batch", etc.).
                Default: "instance".
            conv_block (bool): Whether to use standard convolutional blocks in the decoder path
                (in `UnetrPrUpBlock`). Default: True.
            res_block (bool): Whether to use residual connections within the decoder's
                convolutional layers (`UnetrBasicBlock` and `UnetrUpBlock`). Default: True.
            dropout_rate (float): Dropout rate applied in the ViT backbone and intermediate layers.
                *Effect:* Regularization strength. Must be between 0 and 1. Default: 0.0.
            name (str): Model name. Default: "UNETR".
            **kwargs: Additional keyword arguments passed to `keras.Model`.
        """

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        encoder, input_shape = resolve_encoder(
            encoder=encoder,
            encoder_name=encoder_name,
            input_shape=input_shape,
            allowed_families=UNETR.ALLOWED_BACKBONE_FAMILIES,
            use_class_token=False,
            pooling=None,
        )
        *image_size, _ = input_shape

        patch_dims = encoder.patch_size
        if isinstance(patch_dims, int):
            patch_dims = (patch_dims,) * len(image_size)

        feat_size = tuple(img_d // patch_d for img_d, patch_d in zip(image_size, patch_dims))

        # Get intermediate vectores
        pyramid_outputs = encoder.pyramid_outputs
        num_encoder_blocks = getattr(encoder, "num_layers", 12)
        encoder_depth = 3
        required_keys = self.calculate_skip_keys(num_encoder_blocks, encoder_depth)

        missing_keys = set(required_keys) - set(pyramid_outputs.keys())
        if missing_keys:
            raise ValueError(
                f"The encoder's `pyramid_outputs` is missing one or more required keys. "
                f"Missing keys: {missing_keys}. "
                f"Required: {set(required_keys)}, Available: {set(pyramid_outputs.keys())}"
            )

        # catch input and intermediate feature vectors
        inputs = encoder.input
        skips = [pyramid_outputs[key] for key in required_keys]

        # UNETR Decoder
        decoder_head = self.build_decoder(
            num_classes=num_classes,
            feature_size=feature_size,
            hidden_size=encoder.hidden_dim,
            feat_size=feat_size,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            classifier_activation=classifier_activation,
        )
        last_output = encoder.output

        outputs = decoder_head([inputs] + skips + [last_output])
        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        # === Save config ===
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
        self.feature_size = feature_size
        self.norm_name = norm_name
        self.conv_block = conv_block
        self.res_block = res_block
        self.encoder_name = encoder_name
        self.encoder = encoder
        self.dropout_rate = dropout_rate

    def build_decoder(
        self,
        num_classes,
        feature_size,
        hidden_size,
        feat_size,
        norm_name,
        conv_block,
        res_block,
        classifier_activation=None,
    ):
        def proj_feat(x, hidden_size, feat_size):
            new_shape = (-1, *feat_size, hidden_size)
            return ops.reshape(x, new_shape)

        def apply(inputs):
            enc_input = inputs[0]
            x2, x3, x4, last_output = inputs[1:]

            # Encoder path
            enc1 = UnetrBasicBlock(
                spatial_dims=len(feat_size),
                out_channels=feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )(enc_input)

            enc2 = UnetrPrUpBlock(
                spatial_dims=len(feat_size),
                out_channels=feature_size * 2,
                num_layer=2,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                conv_block=conv_block,
                res_block=res_block,
            )(proj_feat(x2, hidden_size, feat_size))

            enc3 = UnetrPrUpBlock(
                spatial_dims=len(feat_size),
                out_channels=feature_size * 4,
                num_layer=1,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                conv_block=conv_block,
                res_block=res_block,
            )(proj_feat(x3, hidden_size, feat_size))

            enc4 = UnetrPrUpBlock(
                spatial_dims=len(feat_size),
                out_channels=feature_size * 8,
                num_layer=0,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                conv_block=conv_block,
                res_block=res_block,
            )(proj_feat(x4, hidden_size, feat_size))

            # Decoder path
            dec4 = proj_feat(last_output, hidden_size, feat_size)
            dec3 = UnetrUpBlock(
                spatial_dims=len(feat_size),
                out_channels=feature_size * 8,
                kernel_size=3,
                upsample_kernel_size=2,
                res_block=res_block,
            )(dec4, enc4)
            dec2 = UnetrUpBlock(
                spatial_dims=len(feat_size),
                out_channels=feature_size * 4,
                kernel_size=3,
                upsample_kernel_size=2,
                res_block=res_block,
            )(dec3, enc3)
            dec1 = UnetrUpBlock(
                spatial_dims=len(feat_size),
                out_channels=feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                res_block=res_block,
            )(dec2, enc2)
            out = UnetrUpBlock(
                spatial_dims=len(feat_size),
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                res_block=res_block,
            )(dec1, enc1)

            return UnetOutBlock(
                spatial_dims=len(feat_size),
                num_classes=num_classes,
                activation=classifier_activation,
            )(out)

        return apply

    @staticmethod
    def calculate_skip_keys(num_layers, encoder_depth):
        required_keys = []
        for i in range(1, encoder_depth + 1):
            # 1. Determine the relative depth for the skip connection.
            # For num_skips=3, this calculates the fractions: 1/4, 2/4 (1/2), and 3/4.
            fraction = i / (encoder_depth + 1)  # (1/4, 2/4, 3/4)

            # 2. Calculate the corresponding transformer BLOCK INDEX (0 to N-1).
            # This samples the encoder outputs at 25%, 50%, and 75% of the total N blocks.
            # The result (e.g., 3, 6, 9 for N=12) matches the explicit indices used in MONAI UNETR.
            target_block_index = max(0, round(num_layers * fraction))

            # 3. Map the block index to the P-key index in our settings.
            # Our P-key naming starts at P1 (Patch Embedding) and P2 is the output after block 0.
            # P_index = (target_block_index) + (Index offset from Patch Embedding)
            # The block index 3 corresponds to P5 (3 + 2), 6 to P8, and 9 to P11.
            p_index = target_block_index + 2
            required_keys.append(f"P{p_index}")
        return required_keys

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "encoder_name": self.encoder_name,
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "feature_size": self.feature_size,
            "norm_name": self.norm_name,
            "conv_block": self.conv_block,
            "res_block": self.res_block,
            "dropout_rate": self.dropout_rate,
        }
        if self.encoder_name is None and self.encoder is not None:
            config.update({"encoder": keras.saving.serialize_keras_object(self.encoder)})
        return config

    @classmethod
    def from_config(cls, config):
        if "encoder" in config and isinstance(config["encoder"], dict):
            config["encoder"] = keras.layers.deserialize(config["encoder"])
        return super().from_config(config)
