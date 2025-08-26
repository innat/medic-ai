import keras
from keras import ops

from medicai.blocks import UnetOutBlock, UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from medicai.models.vit.vit_backbone import ViTBackbone


class UNETR(keras.Model):
    def __init__(
        self,
        *,
        input_shape,
        num_classes: int,
        classifier_activation=None,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        patch_size: int = 16,
        pos_embed: str = "conv",
        norm_name: str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        name: str = "UNETR",
        **kwargs,
    ):
        """UNETR: U-Net with a Vision Transformer (ViT) backbone for 3D/2D medical image segmentation.

        UNETR integrates a ViT encoder as the backbone with a UNet-style decoder, using 
        projection upsampling blocks and skip connections from intermediate transformer layers.

        Args:
            input_shape (tuple): Shape of the input tensor excluding batch size.
                                For example, (height, width, channels) for 2D
                                or (depth, height, width, channels) for 3D.
            num_classes (int): Number of output segmentation classes.
            classifier_activation (str, optional): Activation function applied to the output layer.
            feature_size (int): Base number of feature channels in decoder blocks.
            hidden_size (int): Hidden size of the transformer encoder.
            mlp_dim (int): Hidden size of MLPs in transformer blocks.
            num_heads (int): Number of attention heads per transformer layer.
            num_layers (int): Number of transformer encoder layers.
            patch_size (int): Size of the patches extracted from input.
            pos_embed (str): Type of positional embedding ("conv" or "learned").
            norm_name (str): Type of normalization for decoder blocks ("instance", "batch", etc.).
            conv_block (bool): Whether to use convolutional blocks in decoder.
            res_block (bool): Whether to use residual blocks in decoder.
            dropout_rate (float): Dropout rate applied in backbone and intermediate layers.
            name (str): Model name.
            **kwargs: Additional keyword arguments passed to keras.Model.

        Example:
            # 3D UNETR for 3-class segmentation
            model = UNETR(
                input_shape=(16, 128, 128, 1),
                num_classes=3,
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                num_layers=12,
                patch_size=16,
                norm_name="instance",
                conv_block=True,
                res_block=True,
                dropout_rate=0.1,
            )

            # Forward pass
            output = model(tf.random.normal((1, 16, 128, 128, 1)))
        """

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        *image_size, num_channels = input_shape
        feat_size = tuple(img_d // patch_size for img_d in image_size)

        # === Backbone ===
        vit_backbone = ViTBackbone(
            input_shape=input_shape,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_size,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            attention_dropout=0.0,
            layer_norm_epsilon=1e-6,
            use_mha_bias=True,
            use_mlp_bias=True,
            use_class_token=False,
            use_patch_bias=True,
            name=f"{name}_backbone",
        )
        inputs = vit_backbone.input
        skips = [
            vit_backbone.get_layer("vit_feature3").output,
            vit_backbone.get_layer("vit_feature6").output,
            vit_backbone.get_layer("vit_feature9").output,
        ]

        # === Decoder ===
        decoder_head = self.build_decoder(
            num_classes=num_classes,
            feature_size=feature_size,
            hidden_size=hidden_size,
            feat_size=feat_size,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            classifier_activation=classifier_activation,
        )
        last_output = vit_backbone.output

        outputs = decoder_head([inputs] + skips + [last_output])
        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        # === Save config ===
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.norm_name = norm_name
        self.conv_block = conv_block
        self.res_block = res_block
        self.dropout_rate = dropout_rate
        self.pos_embed = pos_embed

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

    def get_config(self):
        return {
            "input_shape": self.input_shape[1:],
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "feature_size": self.feature_size,
            "hidden_size": self.hidden_size,
            "mlp_dim": self.mlp_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "patch_size": self.patch_size,
            "norm_name": self.norm_name,
            "conv_block": self.conv_block,
            "res_block": self.res_block,
            "dropout_rate": self.dropout_rate,
            "pos_embed": self.pos_embed,
        }
