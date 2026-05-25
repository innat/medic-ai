import keras

from medicai.utils import DescribeMixin, registration

from .unet import UNet


@keras.saving.register_keras_serializable(package="unet")
@registration.register(name="attention_unet", type="segmentation")
class AttentionUNet(UNet, DescribeMixin):
    """
    AttentionUNet can be constructed either from a registered encoder name or
    from a pre-built encoder instance. The encoder must expose a
    ``pyramid_outputs`` dictionary so the decoder can retrieve the bottleneck
    feature and the skip features used during upsampling.

    The model combines three components:

    1. An encoder that extracts a multi-scale feature pyramid.
    2. A decoder that progressively upsamples the deepest selected feature.
    3. Attention-gated skip connections that filter encoder responses before
       they are fused into the decoder.

    Compared with standard ``UNet``, this model inserts **attention gates** along
    the decoder skip pathways. These gates suppress irrelevant encoder
    responses and help the model focus on spatial regions that are most useful
    for segmentation.

    Args:
        encoder: Optional pre-built Keras model to use as the encoder. It must
            expose ``pyramid_outputs`` with the required feature levels.
        encoder_name: Optional name of a registered backbone model to build and
            use as the encoder.
        input_shape: Optional input shape excluding the batch dimension.
            Required when ``encoder_name`` is used. This can describe either
            2D or 3D inputs.
        encoder_depth: Number of encoder pyramid levels to use. Valid values
            are ``3``, ``4``, and ``5``.
        decoder_block_type: Decoder upsampling strategy. Supported values are
            ``"upsampling"`` and ``"transpose"``.
        decoder_filters: Sequence of channel widths used by the decoder
            refinement path.
        decoder_use_batchnorm: Whether to use batch normalization in decoder
            blocks.
        num_classes: Number of segmentation classes. Must be greater than
            zero.
        classifier_activation: Activation function used by the final
            segmentation head.
        name: Optional model name.
        **kwargs: Additional keyword arguments passed to ``keras.Model``.

    Examples:
        .. code-block:: python

            import tensorflow as tf
            from medicai.models import AttentionUNet

            model = AttentionUNet(
                input_shape=(96, 96, 96, 1),
                num_classes=1,
                encoder_name="efficientnet_b0",
            )

            x = tf.random.uniform(shape=[1, 96, 96, 96, 1])
            y = model(x)
            print(y.shape) # (1, 96, 96, 96, 1)

    .. rubric:: Encoder depth
       :class: api-subheading

    The ``encoder_depth`` argument controls which encoder feature is used as
    the bottleneck and which shallower features are used as skip connections:

    - ``encoder_depth=5`` uses ``P5`` as the bottleneck and ``P4`` through
      ``P1`` as skip connections.
    - ``encoder_depth=4`` uses ``P4`` as the bottleneck and ``P3`` through
      ``P1`` as skip connections.
    - ``encoder_depth=3`` uses ``P3`` as the bottleneck and ``P2`` through
      ``P1`` as skip connections.

    Example:
        Reduce encoder depth::

            import tensorflow as tf
            from medicai.models import AttentionUNet

            model = AttentionUNet(
                input_shape=(96, 96, 96, 1),
                num_classes=1,
                encoder_depth=4,
                encoder_name="efficientnet_b0",
            )

            x = tf.random.uniform(shape=[1, 96, 96, 96, 1])
            y = model(x)
            print(y.shape) # (1, 96, 96, 96, 1)

    .. rubric:: Decoder strategy
       :class: api-subheading

    The decoder can be built with one of two upsampling strategies:

    - ``decoder_block_type="upsampling"`` uses interpolation-based upsampling.
    - ``decoder_block_type="transpose"`` uses trainable transposed
      convolutions.

    Example:
        Use transposed convolution in the decoder::

            import tensorflow as tf
            from medicai.models import AttentionUNet

            model = AttentionUNet(
                input_shape=(96, 96, 96, 1),
                num_classes=1,
                decoder_block_type="transpose",
                encoder_name="efficientnet_b0",
            )

            x = tf.random.uniform(shape=[1, 96, 96, 96, 1])
            y = model(x)
            print(y.shape) # (1, 96, 96, 96, 1)

    .. rubric:: Custom encoder
       :class: api-subheading

    When providing a custom encoder through ``encoder``, ensure that:

    1. It defines a ``pyramid_outputs`` dictionary with the required feature
       levels.
    2. The selected ``encoder_depth`` matches the available pyramid levels.
    3. The feature hierarchy is compatible with the AttentionUNet decoder.

    Example:
        Build the model from a custom encoder::

            import tensorflow as tf
            from medicai.models import AttentionUNet, DenseNetBackbone

            encoder = DenseNetBackbone(
                blocks=[6, 12, 64, 48],
                input_shape=(96, 96, 96, 4),
            )
            model = AttentionUNet(
                encoder=encoder,
                encoder_depth=4,
                num_classes=3,
                classifier_activation="softmax",
                name='densenet_attention_unet'
            )

            x = tf.random.uniform(shape=[1, 96, 96, 96, 4])
            y = model(x)
            print(y.shape) # (1, 96, 96, 96, 3)

    Returns:
        A ``keras.Model`` whose forward pass returns a segmentation tensor of
        shape ``(batch_size, ..., num_classes)``.

    References:
        - Attention U-Net: Learning Where to Look for the Pancreas.
          `arXiv:1804.03999 <https://arxiv.org/abs/1804.03999>`_
    """

    def __init__(
        self,
        *,
        input_shape=None,
        encoder_name=None,
        encoder=None,
        encoder_depth=5,
        decoder_block_type="upsampling",
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
        num_classes=1,
        classifier_activation="sigmoid",
        name=None,
        **kwargs,
    ):
        # decoder_attentio_gate: A boolean indicating whether to use attention
        # blocks in the decoder.
        self.decoder_attention_gate = True
        super().__init__(
            input_shape=input_shape,
            encoder_name=encoder_name,
            encoder=encoder,
            encoder_depth=encoder_depth,
            decoder_block_type=decoder_block_type,
            decoder_filters=decoder_filters,
            decoder_use_batchnorm=decoder_use_batchnorm,
            num_classes=num_classes,
            classifier_activation=classifier_activation,
            name=name or f"AttentionUNet{len(input_shape) - 1}D",
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"decoder_attention_gate": True})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
