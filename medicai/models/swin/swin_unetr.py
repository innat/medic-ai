import keras

from medicai.blocks import UNetOutBlock, UNETRBasicBlock, UNETRUpsamplingBlock
from medicai.utils import DescribeMixin, registration, resolve_encoder, validate_activation


@keras.saving.register_keras_serializable(package="swin.unetr")
@registration.register(name="swin_unetr", type="segmentation")
class SwinUNETR(keras.Model, DescribeMixin):
    """
    SwinUNETR is a semantic segmentation model built from a Swin Transformer
    encoder and a UNETR-style decoder.

    This implementation supports both 2D and 3D input shapes. The encoder
    provides a hierarchical feature pyramid, and the decoder refines these
    features with convolutional blocks, progressive upsampling, and skip
    connections to produce a dense segmentation map.

    The model can be constructed either from a registered Swin encoder name or
    from a pre-built encoder instance. The encoder must expose a
    ``pyramid_outputs`` dictionary containing ``P1`` through ``P5``.

    The decoder uses the feature hierarchy as follows:

    1. The raw input is processed by a shallow convolution block to form the
       highest-resolution skip branch.
    2. ``P1``, ``P2``, and ``P3`` are refined and used as intermediate skip
       connections.
    3. ``P5`` is used as the bottleneck feature.
    4. ``P4`` is fused during the first decoder upsampling stage, and the
       remaining refined features are fused in later decoding stages until the
       final segmentation output is produced.

    By default, this model resolves named encoders with the
    ``swin_unetr_like`` downsampling strategy, which omits the final
    patch-merging step used in standard Swin classification backbones. This
    matches the feature hierarchy expected by SwinUNETR.

    When ``stage_wise_conv=True``, the encoder inserts an additional residual
    convolution block at the start of each Swin stage. This follows the
    SwinUNETR-V2 style of stage-wise convolutional refinement and can be used
    with either Swin V1 or Swin V2 encoders.

    When providing a custom encoder through ``encoder``, ensure that:

    1. It defines a ``pyramid_outputs`` dictionary with keys ``P1`` through
       ``P5``.
    2. Its feature hierarchy matches the SwinUNETR decoder expectations.
    3. It is built with a compatible downsampling pattern, typically
       ``downsampling_strategy="swin_unetr_like"`` for Swin-based encoders.

    Args:
        input_shape: Optional input shape excluding the batch dimension.
            Required when ``encoder_name`` is used. This can describe either
            2D or 3D inputs.
        encoder_name: Optional name of a registered Swin backbone to build and
            use as the encoder.
        encoder: Optional pre-built Keras model to use as the encoder. It must
            expose ``pyramid_outputs`` with the required feature levels.
        num_classes: Number of segmentation classes. Must be greater than
            zero.
        patch_size: Patch size used when building a named Swin encoder.
        window_size: Window size used by shifted-window self-attention when
            building a named Swin encoder.
        classifier_activation: Activation function used by the final
            segmentation head.
        feature_size: Base channel width used by the decoder blocks.
        res_block: Whether residual blocks are used inside decoder refinement
            blocks.
        norm_name: Normalization type used in decoder blocks.
        stage_wise_conv: Whether to enable stage-wise convolutional refinement
            in the encoder, following the SwinUNETR-V2 style.
        name: Optional model name.
        **kwargs: Additional keyword arguments passed to ``keras.Model``.

    Returns:
        A ``keras.Model`` whose forward pass returns a segmentation tensor of
        shape ``(batch_size, ..., num_classes)`` at the model output
        resolution.

    Examples:
        Build ``SwinUNETR`` from a registered Swin V1 encoder::

            import tensorflow as tf
            from medicai.models import SwinUNETR

            model = SwinUNETR(
                encoder_name="swin_tiny",
                input_shape=(96, 96, 96, 4),
                num_classes=3,
                classifier_activation="sigmoid",
            )

            x = tf.random.uniform(shape=[1,96,96,96,4])
            y = model(x)
            print(y.shape) # (1, 96, 96, 96, 3)

        Build ``SwinUNETR`` from a registered Swin V2 encoder::

            import tensorflow as tf
            from medicai.models import SwinUNETR

            model = SwinUNETR(
                encoder_name="swin_tiny_v2",
                input_shape=(96, 96, 96, 4),
                num_classes=3,
                classifier_activation="sigmoid",
            )

            x = tf.random.uniform(shape=[1,96,96,96,4])
            y = model(x)
            print(y.shape) # (1, 96, 96, 96, 3)

        Build ``SwinUNETR-V2`` with stage-wise convolutional refinement::

            import tensorflow as tf
            from medicai.models import SwinUNETR

            model = SwinUNETR(
                encoder_name="swin_tiny",
                input_shape=(96, 96, 96, 4),
                num_classes=1,
                stage_wise_conv=True,
                classifier_activation="sigmoid",
            )

            x = tf.random.uniform(shape=[1,96,96,96,4])
            y = model(x)
            print(y.shape) # (1, 96, 96, 96, 3)

        Build ``SwinUNETR`` from a custom encoder::

            import tensorflow as tf
            from medicai.models import SwinBackboneV2, SwinUNETR

            custom_encoder = SwinBackboneV2(
                input_shape=(64, 128, 128, 1),
                embed_dim=48,
                window_size=8,
                patch_size=2,
                downsampling_strategy="swin_unetr_like",
            )
            model = SwinUNETR(encoder=custom_encoder)

            x = tf.random.uniform(shape=[1, 64, 128, 128, 1])
            y = model(x)
            print(y.shape) # (1, 64, 128, 128, 1)

    References:
        - Swin UNETR: Swin Transformers for Semantic Segmentation of Brain
          Tumors in MRI Images. https://arxiv.org/abs/2201.01266
        - SwinUNETR-V2: Stronger Swin Transformers with Stagewise Convolutions
          for 3D Medical Image Segmentation.
          https://link.springer.com/chapter/10.1007/978-3-031-43901-8_40
    """

    ALLOWED_BACKBONE_FAMILIES = ["swin"]

    def __init__(
        self,
        *,
        input_shape=None,
        encoder_name=None,
        encoder=None,
        num_classes=1,
        patch_size=2,
        window_size=7,
        classifier_activation=None,
        feature_size=48,
        res_block=True,
        norm_name="instance",
        stage_wise_conv=False,
        name=None,
        **kwargs,
    ):
        """
        Initializes the SwinUNETR model.

        Args:
            input_shape (tuple): The shape of the input tensor. Must be 4D (H, W, C) for 2D or
                5D (D, H, W, C) for 3D inputs. The batch dimension is excluded.
            encoder_name (str, optional): Name of the Swin Transformer backbone preset (e.g., 'swin_tiny').
                Required if 'encoder' is None.
            encoder (keras.Model, optional): A pre-instantiated Swin Transformer encoder model.
                If provided, 'encoder_name' is ignored.
            num_classes (int): The number of segmentation classes. Default is 1.
            patch_size (int or tuple): Size of the non-overlapping patches used by the Swin Encoder.
                Can be a single int or a tuple (PD, PH, PW) or (PH, PW). Default is 2.
            window_size (int or tuple): Size of the attention windows (WD, WH, WW) or (WH, WW). Default is 7.
            classifier_activation (str, optional): The activation function for the final
                classification layer (e.g., 'softmax', 'sigmoid'). If None, no activation is applied.
                Default is None.
            feature_size (int): The base feature map size in the decoder. The decoder channels
                will be scaled relative to this. Default is 48.
            res_block (bool): Whether to use residual connections in the decoder blocks. Default is True.
            norm_name (str): The type of normalization to use in the decoder blocks
                (e.g., 'instance', 'batch'). Default is "instance".
            stage_wise_conv (bool): If True, a convolutional layer is used to adjust channel dimensions
                at the start of each Swin stage, matching the original SwinUNETR-V2 structure. Default is False.
            name (str, optional): Name of the model. Defaults to "SwinUNETR2D" or "SwinUNETR3D".
            **kwargs: Additional keyword arguments passed to the base Model class.

        Raises:
            ValueError: If the encoder does not provide the required pyramid outputs (P1 to P5).
        """

        # Compute spatial dimention and resolve encoder arguments
        encoder, input_shape = resolve_encoder(
            encoder=encoder,
            encoder_name=encoder_name,
            input_shape=input_shape,
            patch_size=patch_size,
            window_size=window_size,
            allowed_families=SwinUNETR.ALLOWED_BACKBONE_FAMILIES,
            downsampling_strategy="swin_unetr_like",
            stage_wise_conv=stage_wise_conv,
            pooling=None,
        )

        # number of classes must be positive.
        if num_classes <= 0:
            raise ValueError(
                f"Number of classes (`num_classes`) must be greater than 0, "
                f"but received {num_classes}."
            )

        # verify input activation.
        classifier_activation = validate_activation(classifier_activation)

        # get spatial dimention
        spatial_dims = len(input_shape) - 1

        # Get intermediate vectores
        pyramid_outputs = encoder.pyramid_outputs
        required_keys = ["P1", "P2", "P3", "P4", "P5"]
        missing_keys = set(required_keys) - set(pyramid_outputs.keys())
        if missing_keys:
            raise ValueError(
                f"The encoder's `pyramid_outputs` is missing one or more required keys. "
                f"Missing keys: {missing_keys}. "
                f"Required: {set(required_keys)}, Available: {set(pyramid_outputs.keys())}"
            )

        inputs = encoder.input
        skips = [pyramid_outputs[key] for key in required_keys]
        unetr_head = self.build_decoder(
            num_classes=num_classes,
            feature_size=feature_size,
            res_block=True,
            norm_name=norm_name,
            classifier_activation=classifier_activation,
        )

        # Combine encoder and decoder
        outputs = unetr_head([inputs] + skips)
        super().__init__(
            inputs=inputs, outputs=outputs, name=name or f"SwinUNETR{spatial_dims}D", **kwargs
        )

        self.num_classes = num_classes
        self.patch_size = patch_size
        self.window_size = window_size
        self.feature_size = feature_size
        self.res_block = res_block
        self.norm_name = norm_name
        self.encoder_name = encoder_name
        self.encoder = encoder
        self.classifier_activation = classifier_activation

    def build_decoder(
        self,
        num_classes=4,
        feature_size=16,
        res_block=True,
        norm_name="instance",
        classifier_activation=None,
    ):
        """Builds the UNETR-like decoder part of the model.

        Args:
            num_classes (int): The number of segmentation classes. Default is 4.
            feature_size (int): The base feature map size in the decoder. Default is 16.
            res_block (bool): Whether to use residual connections in the decoder blocks.
                Default is True.
            norm_name (str): The type of normalization to use in the decoder blocks
                (e.g., 'instance', 'batch'). Default is "instance".
            classifier_activation (str, optional): The activation function for the final
                classification layer. Default is None.

        Returns:
            callable: A function that takes a list of encoder outputs (including input)
                and skip connections and returns the final segmentation logits.
        """

        def apply(inputs):
            enc_input = inputs[0]
            hidden_states_out = inputs[1:]

            # Encoder 1 (process raw input)
            enc0 = UNETRBasicBlock(
                filters=feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
                name="decoder_input_cnn_stem",
            )(enc_input)

            # Encoder 2 (process hidden_states_out[0])
            enc1 = UNETRBasicBlock(
                filters=feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
                name="swin_stage1_refine",
            )(hidden_states_out[0])

            # Encoder 3 (process hidden_states_out[1])
            enc2 = UNETRBasicBlock(
                filters=feature_size * 2,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
                name="swin_stage2_refine",
            )(hidden_states_out[1])

            # Encoder 4 (process hidden_states_out[2])
            enc3 = UNETRBasicBlock(
                filters=feature_size * 4,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
                name="swin_stage3_refine",
            )(hidden_states_out[2])

            # Encoder 5 (process hidden_states_out[4] as bottleneck)
            dec4 = UNETRBasicBlock(
                filters=feature_size * 16,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
                name="swin_decoder_bottleneck",
            )(hidden_states_out[4])

            # Decoder 5 (upsample dec4 and concatenate with hidden_states_out[3])
            dec3 = UNETRUpsamplingBlock(
                filters=feature_size * 8,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                name="swin_decoder_up_stage3",
            )([dec4, hidden_states_out[3]])

            # Decoder 4 (upsample dec3 and concatenate with enc3)
            dec2 = UNETRUpsamplingBlock(
                filters=feature_size * 4,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                name="swin_decoder_up_stage2",
            )([dec3, enc3])

            # Decoder 3 (upsample dec2 and concatenate with enc2)
            dec1 = UNETRUpsamplingBlock(
                filters=feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                name="swin_decoder_up_stage1",
            )([dec2, enc2])

            # Decoder 2 (upsample dec1 and concatenate with enc1)
            dec0 = UNETRUpsamplingBlock(
                filters=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                name="swin_decoder_up_stage0",
            )([dec1, enc1])

            out = UNETRUpsamplingBlock(
                filters=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                name="swin_decoder_output_stage",
            )([dec0, enc0])

            # Final output (process dec0 and produce logits)
            logits = UNetOutBlock(
                num_classes,
                activation=classifier_activation,
                name="swinunetr_segmentation_head",
            )(out)
            return logits

        return apply

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "encoder_name": self.encoder_name,
            "num_classes": self.num_classes,
            "patch_size": self.patch_size,
            "window_size": self.window_size,
            "classifier_activation": self.classifier_activation,
            "feature_size": self.feature_size,
            "res_block": self.res_block,
            "norm_name": self.norm_name,
        }

        if self.encoder_name is None and self.encoder is not None:
            config.update({"encoder": keras.saving.serialize_keras_object(self.encoder)})

        return config

    @classmethod
    def from_config(cls, config):
        if "encoder" in config and isinstance(config["encoder"], dict):
            config["encoder"] = keras.layers.deserialize(config["encoder"])
        return super().from_config(config)
