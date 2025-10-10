import keras

from medicai.utils import DescribeMixin

from .unet import UNet


@keras.saving.register_keras_serializable(package="unet")
class AttentionUNet(UNet, DescribeMixin):
    """
    Attention U-Net model for semantic segmentation.

    Extends the UNet architecture by integrating Attention Gates in the decoder path,
    allowing the network to focus on relevant spatial regions.

    Reference:
        Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas"
        (https://arxiv.org/abs/1804.03999)

    Example:
    >>> from medicai.models import AttentionUNet
    >>> model = AttentionUNet(input_shape=(96, 96, 1), encoder_name="densenet121")
    >>> model = AttentionUNet(input_shape=(96, 96, 1), encoder_name="densenet121", encoder_depth=4)
    >>> model = AttentionUNet(input_shape=(96, 96, 96, 1), encoder_name="efficientnet_b0")
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
