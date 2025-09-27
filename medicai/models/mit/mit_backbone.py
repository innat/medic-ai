import keras
import numpy as np
from keras import ops

from medicai.utils import DescribeMixin, get_norm_layer, parse_model_inputs

from .mit_layers import HierarchicalTransformerEncoder, OverlappingPatchingAndEmbedding


@keras.utils.register_keras_serializable(package="mit.backbone")
class MiTBackbone(keras.Model, DescribeMixin):
    """MixVisionTransformer (MixViT) Model.

    This class implements the encoder backbone of the SegFormer architecture. It is a
    hierarchical vision transformer that processes input data (2D images or 3D volumes)
    through multiple stages. Each stage consists of an overlapping patch embedding layer,
    followed by a series of efficient transformer encoder blocks. The use of overlapping
    patches and spatially reduced attention makes the model efficient for high-resolution
    inputs while capturing both local and global features.

    The model is built using the Keras Functional API, with a progressive
    downsampling of the spatial dimensions and an increase in the feature dimensions,
    similar to a convolutional neural network.

    ## Key Features and Strengths:
    1.  **Efficient Self-Attention:** MiT replaces the standard vision transformer's
        (ViT) global self-attention with a **Spatially-Reduced Attention (SRA)**
        mechanism. SRA significantly reduces the computational cost from quadratic
        to near-linear with respect to image size, allowing it to process high-resolution
        inputs efficiently.
    2.  **Multi-Scale Feature Pyramid:** It generates a multi-scale feature pyramid,
        similar to Convolutional Neural Networks (CNNs), by using four stages with
        gradually increasing receptive fields. This is achieved through overlapping
        patch embedding and successive attention blocks, making it ideal as a
        backbone for downstream tasks (like the UNet or segmentation models).
    3.  **Lightweight and Configurable:** The architecture is highly parameterizable,
        allowing for the creation of various models (e.g., MiT-B0 to MiT-B5) by
        adjusting parameters like embedding dimensions, depth, and attention heads,
        balancing performance and model size.

    Example:
    >>> from medicai.models import MiTBackbone
    >>> # 2D Model (e.g., for ImageNet)
    >>> model_2d = MiTBackbone(input_shape=(224, 224, 3), project_dim=[32, 64, 160, 256], ...)
    >>> # 3D Model (e.g., for medical volumes)
    >>> model_3d = MiTBackbone(input_shape=(64, 64, 64, 1), project_dim=[32, 64, 160, 256], ...)

    Reference:
        https://github.com/keras-team/keras-hub

    """

    def __init__(
        self,
        input_shape,
        include_rescaling=False,
        max_drop_path_rate=0.1,
        layer_norm_epsilon=1e-5,
        qkv_bias=True,
        project_dim=[32, 64, 160, 256],
        sr_ratios=[4, 2, 1, 1],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        depths=[2, 2, 2, 2],
        mlp_ratios=[4, 4, 4, 4],
        name=None,
        **kwargs,
    ):
        """
        Initializes the MiTBackbone model.

        Args:
            input_shape (tuple): The shape of the input data, excluding the batch dimension.
            include_rescaling: A boolean indicating whether to include a
                `Rescaling` layer at the beginning of the model. If `True`,
                the input pixels will be scaled from `[0, 255]` to `[0, 1]`. Defaults to False.
            max_drop_path_rate (float, optional): The maximum rate for stochastic depth.
                The dropout rate is linearly increased across all transformer blocks. Defaults to 0.1.
            layer_norm_epsilon (float, optional): A small value for numerical stability in layer normalization.
                Defaults to 1e-5.
            qkv_bias (bool): A boolean flag to indicate applying bias to projected queries, keys, and values.
            project_dim (list[int], optional): A list of feature dimensions for each stage.
                Defaults to [32, 64, 160, 256].
            sr_ratios (list[int], optional): A list of spatial reduction ratios for each stage's
                attention layers. Defaults to [4, 2, 1, 1].
            patch_sizes (list[int], optional): A list of patch sizes for the embedding layer
                in each stage. Defaults to [7, 3, 3, 3].
            strides (list[int], optional): A list of strides for the embedding layer in each stage.
                Defaults to [4, 2, 2, 2].
            num_heads (list[int], optional): A list of the number of attention heads for each stage.
                Defaults to [1, 2, 5, 8].
            depths (list[int], optional): A list of the number of transformer blocks for each stage.
                Defaults to [2, 2, 2, 2].
            mlp_ratios (list[int], optional): A list of MLP expansion ratios for each stage.
                Defaults to [4, 4, 4, 4].
            name (str, optional): The name of the model. Defaults to None.
            **kwargs: Standard Keras Model keyword arguments.
        """
        spatial_dims = len(input_shape) - 1
        num_layers = len(depths)

        # Create a list of linearly increasing drop path rate
        dpr = [x for x in np.linspace(0.0, max_drop_path_rate, sum(depths))]

        # initialize model input
        input = parse_model_inputs(input_shape=input_shape, name="mixvit_input")
        x = input
        cur = 0
        pyramid_outputs = {}

        if include_rescaling:
            x = keras.layers.Rescaling(1.0 / 255)(x)

        # Loop through each hierarchical stage of the model.
        for i in range(num_layers):
            # Overlapping Patch Embedding Stage
            patch_embed = OverlappingPatchingAndEmbedding(
                project_dim=project_dim[i],
                patch_size=patch_sizes[i],
                stride=strides[i],
                name=f"overlap_patch_and_embed_{i}",
            )
            x = patch_embed(x)

            # Transformer Blocks
            for k in range(depths[i]):
                x = HierarchicalTransformerEncoder(
                    project_dim=project_dim[i],
                    num_heads=num_heads[i],
                    sr_ratio=sr_ratios[i],
                    mlp_ratio=mlp_ratios[i],
                    drop_prob=dpr[cur + k],
                    qkv_bias=qkv_bias,
                    layer_norm_epsilon=layer_norm_epsilon,
                    spatial_dims=spatial_dims,
                    name=f"hierarchical_encoder_{i}_{k}",
                )(x)
            cur += depths[i]

            # Layer Normalization
            x = get_norm_layer(norm_name="layer", epsilon=layer_norm_epsilon)(x)

            # Reshape output to a spatial feature map for the next stage.
            n_patches = ops.shape(x)[1]
            current_spatial_dims = int(ops.round(n_patches ** (1 / spatial_dims)))
            current_spatial_dims = [current_spatial_dims] * spatial_dims
            x = keras.layers.Reshape(current_spatial_dims + [project_dim[i]])(x)
            pyramid_outputs[f"P{i+1}"] = x

        super().__init__(inputs=input, outputs=x, name=name or f"mit{spatial_dims}D", **kwargs)

        self.pyramid_outputs = pyramid_outputs
        self.include_rescaling = include_rescaling
        self.project_dim = project_dim
        self.qkv_bias = qkv_bias
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.num_heads = num_heads
        self.depths = depths
        self.sr_ratios = sr_ratios
        self.max_drop_path_rate = max_drop_path_rate
        self.mlp_ratios = mlp_ratios
        self.layer_norm_epsilon = layer_norm_epsilon

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape[1:],
                "include_rescaling": self.include_rescaling,
                "project_dim": self.project_dim,
                "qkv_bias": self.qkv_bias,
                "patch_sizes": self.patch_sizes,
                "strides": self.strides,
                "num_heads": self.num_heads,
                "depths": self.depths,
                "sr_ratios": self.sr_ratios,
                "mlp_ratios": self.mlp_ratios,
                "max_drop_path_rate": self.max_drop_path_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
