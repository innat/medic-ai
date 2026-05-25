import keras
import numpy as np
from keras import ops

from medicai.utils import DescribeMixin, get_norm_layer, parse_model_inputs

from .mit_layers import HierarchicalTransformerEncoder, OverlappingPatchingAndEmbedding


@keras.utils.register_keras_serializable(package="mit.backbone")
class MiTBackbone(keras.Model, DescribeMixin):
    """
    MiT feature-extraction backbone with multi-scale feature outputs.

    This class builds only the backbone portion of a Mix Transformer (MiT)
    model. It is intended for workflows that need reusable feature maps rather
    than a final classification layer, such as custom classifiers, detection
    heads, or segmentation decoders.

    The backbone is constructed in the following stages:

    1. An input layer is created from ``input_shape``.
    2. An optional ``Rescaling`` layer normalizes raw image intensities.
    3. Each stage begins with overlapping patch embedding, which downsamples
       the spatial dimensions while projecting the input into a new feature
       space.
    4. A stack of hierarchical transformer encoder blocks is applied at each
       stage using spatial-reduction attention and stage-specific MLP ratios,
       attention heads, and drop path rates.
    5. The sequence output of each stage is layer-normalized, reshaped back
       into a spatial feature map, and stored in ``pyramid_outputs``.

    Args:
        input_shape: A tuple specifying the input shape of the model, not
            including the batch size. This can describe either 2D or 3D
            inputs.
        include_rescaling: A boolean indicating whether to include a
            ``Rescaling`` layer at the beginning of the model.
        max_drop_path_rate: A float specifying the maximum stochastic depth
            rate distributed across the transformer blocks.
        layer_norm_epsilon: A float specifying the epsilon value used in layer
            normalization.
        qkv_bias: A boolean indicating whether to apply bias terms to query,
            key, and value projections.
        project_dim: A list of integers specifying the embedding dimension for
            each stage.
        sr_ratios: A list of integers specifying the spatial-reduction ratio
            used by attention in each stage.
        patch_sizes: A list of integers specifying the patch embedding kernel
            size for each stage.
        strides: A list of integers specifying the patch embedding stride for
            each stage.
        num_heads: A list of integers specifying the number of attention heads
            in each stage.
        depths: A list of integers specifying the number of transformer blocks
            in each stage.
        mlp_ratios: A list of integers specifying the MLP expansion ratio in
            each stage.
        name: (Optional) The name of the model.

    Returns:
        A ``keras.Model`` whose forward pass returns the final backbone
        feature tensor. Intermediate multi-scale features are available in
        the ``pyramid_outputs`` attribute.

    Examples:
        .. code-block:: python

            import torch
            from medicai.models.mit import MiTBackbone

            model = MiTBackbone(
                input_shape=(224, 224, 3),
                project_dim=[32, 64, 160, 256],
                num_heads=[1, 2, 5, 8],
                depths=[2, 2, 2, 2],
                name='mit_backbone'
            )
            x = torch.randn((1, 224, 224, 3))
            y = model(x)
            print(y.shape) # torch.Size([1, 7, 7, 256])


    References:
        - SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. NeurIPS 2021.
            `arXiv:2105.15203 <https://arxiv.org/abs/2105.15203>`_
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
            x = get_norm_layer(layer_type="layer", epsilon=layer_norm_epsilon)(x)

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
