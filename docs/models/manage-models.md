The `medic-ai` library provides state-of-the-art models for 2D and 3D medical image classification and segmentation. It features models based on both **Convolutional Neural Networks** (CNNs) and **Transformers**, which have been translated from their official releases to work with **Keras**. This allows them to function seamlessly across various backends, including **TensorFlow**, **PyTorch**, and **JAX**. The model inputs can be either **3D** `(depth × height × width × channel)` or **2D** `(height × width × channel)`. The following table lists the currently supported models along with their supported input modalities, primary tasks, and underlying architecture type.  

| Model        | Supported Modalities | Primary Task   | Architecture Type         |
| ------------ | -------------------- | -------------- | ------------------------- |
| [DenseNet121](#-densenet121)     | 2D, 3D               | Classification | CNN                       |
| DenseNet169     | 2D, 3D               | Classification | CNN                       |
| DenseNet201     | 2D, 3D               | Classification | CNN                       |
| ViT          | 2D, 3D               | Classification | Transformer               |
 Swin Transformer          | 2D, 3D               | Classification | Transformer               |
| DenseUNet121 | 2D, 3D               | Segmentation   | CNN                       |
| DenseUNet169 | 2D, 3D               | Segmentation   | CNN                       |
| DenseUNet201 | 2D, 3D               | Segmentation   | CNN                       |
| UNETR        | 2D, 3D               | Segmentation   | Transformer               |
| SwinUNETR    | 2D, 3D               | Segmentation   | Transformer               |
| TransUNet    | 2D, 3D               | Segmentation   | Transformer |
| SegFormer    | 2D, 3D               | Segmentation   | Transformer |


All models in `medicai` are flexible and can be built as either **2D** or **3D** models. The library automatically configures the model based on the provided `input_shape` argument. Specifying `(depth, height, width, channel)` creates a **3D** model, whereas passing `(height, width, channel)` builds a **2D** model.


## **DenseNet121**

A 2D or 3D DenseNet-121 model for classification task.

```python
medicai.models.DenseNet121(
    input_shape,
    include_rescaling=False,
    include_top=True,
    num_classes=1000,
    pooling=None,
    classifier_activation="softmax",
    name=None,
)

# Build 3D model.
num_classes = 1
input_shape = (64, 64, 64, 1)
model = medicai.models.DenseNet121(
    input_shape=input_shape, num_classes=num_classes
)

# Build 2D model.
input_shape = (64, 64, 1)
model = medicai.models.DenseNet121(
    input_shape=input_shape, 
    num_classes=num_classes
)
```

**Reference**

- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (CVPR 2017)

**Arguments**

- **include_top**: whether to include the fully-connected layer at the top of the network.
- **input_shape**: Input tensor shape, excluding batch size. It can be either `(depth, height, width, channel)` or `(height, width, channel)`.
- **include_rescaling**: Whether to include input rescaling layer
- **pooling**: Optional pooling mode for feature extraction when `include_top` is `False`.
    - `None` means that the output of the model will be the 4D/5D tensor output of the last convolutional layer.
    - `avg` means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D/3D tensor.
    - `max` means that global max pooling will be applied.
- **num_classes**: Number of classes to classify samples.
- **classifier_activation**: The activation function to use on the top layer.
- **name**: The name of the model.


## **DenseNet169**

A 2D or 3D DenseNet-169 model for classification task.

```python
medicai.models.DenseNet169(
    input_shape,
    include_rescaling=False,
    include_top=True,
    num_classes=1000,
    pooling=None,
    classifier_activation="softmax",
    name=None,
)

# Build 3D model.
num_classes = 1
input_shape = (64, 64, 64, 1)
model = medicai.models.DenseNet169(
    input_shape=input_shape, num_classes=num_classes
)

# Build 2D model.
input_shape = (64, 64, 1)
model = medicai.models.DenseNet169(
    input_shape=input_shape, 
    num_classes=num_classes
)
```

**Reference**

- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (CVPR 2017)

**Arguments**

- **include_top**: whether to include the fully-connected layer at the top of the network.
- **input_shape**: Input tensor shape, excluding batch size. It can be either `(depth, height, width, channel)` or `(height, width, channel)`.
- **include_rescaling**: Whether to include input rescaling layer
- **pooling**: Optional pooling mode for feature extraction when `include_top` is `False`.
    - `None` means that the output of the model will be the 4D/5D tensor output of the last convolutional layer.
    - `avg` means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D/3D tensor.
    - `max` means that global max pooling will be applied.
- **num_classes**: Number of classes to classify samples.
- **classifier_activation**: The activation function to use on the top layer.
- **name**: The name of the model.


## **DenseNet201**

A 2D or 3D DenseNet-201 model for classification task.

```python
medicai.models.DenseNet201(
    input_shape,
    include_rescaling=False,
    include_top=True,
    num_classes=1000,
    pooling=None,
    classifier_activation="softmax",
    name=None,
)

# Build 3D model.
num_classes = 1
input_shape = (64, 64, 64, 1)
model = medicai.models.DenseNet201(
    input_shape=input_shape, num_classes=num_classes
)

# Build 2D model.
input_shape = (64, 64, 1)
model = medicai.models.DenseNet201(
    input_shape=input_shape, 
    num_classes=num_classes
)
```

**Reference**

- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (CVPR 2017)

**Arguments**

- **include_top**: whether to include the fully-connected layer at the top of the network.
- **input_shape**: Input tensor shape, excluding batch size. It can be either `(depth, height, width, channel)` or `(height, width, channel)`.
- **include_rescaling**: Whether to include input rescaling layer
- **pooling**: Optional pooling mode for feature extraction when `include_top` is `False`.
    - `None` means that the output of the model will be the 4D/5D tensor output of the last convolutional layer.
    - `avg` means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D/3D tensor.
    - `max` means that global max pooling will be applied.
- **num_classes**: Number of classes to classify samples.
- **classifier_activation**: The activation function to use on the top layer.
- **name**: The name of the model.


## **Vision Transformer (ViT)**

A 2D and 3D Vision Transformer (ViT) model for classification.

This class implements a Vision Transformer (ViT) model,
supporting both 2D and 3D inputs. The model consists of a ViT backbone,
optional intermediate pre-logits layer, dropout, and a classification head

```python
medicai.models.ViT(
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
)

# Build 3D model.
input_shape = (16, 32, 32, 1)
num_classes = 10
model = medicai.models.ViT(
    input_shape=input_shape, 
    num_classes=num_classes
)

# Build 2D model.
input_shape = (32, 32, 1)
model = medicai.models.ViT(
    input_shape=input_shape, 
    num_classes=num_classes
)
```

**Reference**

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (CVPR 2020)

**Arguments**

- **input_shape (tuple)**: Shape of the input tensor excluding batch size.
    For example, `(height, width, channels)` for 2D
    or `(depth, height, width, channels)` for 3D.
- **num_classes (int)**: Number of output classes for classification.
- **patch_size (int or tuple)**: Size of the patches extracted from the input.
- **num_layers (int)**: Number of transformer encoder layers.
- **num_heads (int)**: Number of attention heads in each transformer layer.
- **hidden_dim (int)**: Hidden dimension size of the transformer encoder.
- **mlp_dim (int)**: Hidden dimension size of the MLP in transformer blocks.
- **pooling (str)**: Pooling strategy for the output. `token` for CLS token,
    `gap` for global average pooling over spatial dimensions.
- **intermediate_dim (int, optional)**: Dimension of optional pre-logits dense layer.
    If `None`, no intermediate layer is used.
- **classifier_activation (str, optional)**: Activation function for the output layer.
- **dropout (float)**: Dropout rate applied before the output layer.
- **name (str)**: Name of the model.


## **Swin Transformer**

A 2D and 3D Swin Transformer model for classification.

This model utilizes the Swin Transformer backbone for feature extraction
from 2D or 3D input data and includes a global average pooling layer followed
by a dense layer for classification.

```python
medicai.models.SwinTransformer(
    input_shape,
    num_classes,
    classifier_activation=None,
    name="swin_transformer",
)

# 3D model.
num_classes = 4
input_shape = (96, 96, 96, 1)
model = medicai.models.SwinTransformer(
    input_shape=input_shape, 
    num_classes=num_classes
)

# 2D model.
num_classes = 4
input_shape = (96, 96, 1)
model = medicai.models.SwinTransformer(
    input_shape=input_shape, 
    num_classes=num_classes
)
```

**Reference**

- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) (CVPR 2021)


**Arguments**

- **input_shape (tuple)**: Shape of the input tensor excluding batch size.
    For example, `(height, width, channels)` for 2D
    or `(depth, height, width, channels)` for 3D.
- **num_classes (int)**: Number of output classes for classification.
- **classifier_activation (str, optional)**: Activation function for the output layer.
- **name (str)**: Name of the model.

## **DenseUNet121**

A UNet model with a DenseNet-121 backbone

This model is a UNet architecture for image segmentation that uses a DenseNet-121 as its feature-extracting encoder. It's built to provide a powerful and flexible solution for both 2D and 3D segmentation tasks. .

```python
medicai.models.DenseUNet121(
    input_shape,
    num_classes,
    classifier_activation=None,
    decoder_block_type="upsampling",
    decoder_filters=(256, 128, 64, 32, 16),
    name='dense_unet_121',
)

# 3D model.
num_classes = 1
input_shape = (64, 64, 64, 1)
model = medicai.models.DenseUNet121(
    input_shape=input_shape,
    num_classes=num_classes
)

# 2D model.
num_classes = 1
input_shape = (64, 64, 1)
model = medicai.models.DenseUNet121(
    input_shape=input_shape, 
    num_classes=num_classes
)
```

**Reference**

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (CVPR 2015)

**Arguments**

- **input_shape (tuple)**: Shape of the input tensor excluding batch size.
    For example, `(height, width, channels)` for 2D
    or `(depth, height, width, channels)` for 3D.
- **num_classes (int)**: Number of output classes for classification.
- **classifier_activation (str, optional)**: Activation function for the output layer.
- **decoder_block_type**: Decoder block type, either `upsampling` or `transpose`.
- **decoder_filters**: The projection filters in decoder blocks. Default: `(256, 128, 64, 32, 16)`.
- **name (str)**: Name of the model.

## **DenseUNet169**

A UNet model with a DenseNet-169 backbone

This model is a UNet architecture for image segmentation that uses a DenseNet-169 as its feature-extracting encoder. It's built to provide a powerful and flexible solution for both 2D and 3D segmentation tasks. .

```python
medicai.models.DenseUNet169(
    input_shape,
    num_classes,
    classifier_activation=None,
    decoder_block_type="upsampling",
    decoder_filters=(256, 128, 64, 32, 16),
    name='dense_unet_121',
)

# 3D model.
num_classes = 1
input_shape = (64, 64, 64, 1)
model = medicai.models.DenseUNet169(
    input_shape=input_shape,
    num_classes=num_classes
)

# 2D model.
num_classes = 1
input_shape = (64, 64, 1)
model = medicai.models.DenseUNet169(
    input_shape=input_shape, 
    num_classes=num_classes
)
```

**Reference**

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (CVPR 2015)

**Arguments**

- **input_shape (tuple)**: Shape of the input tensor excluding batch size.
    For example, `(height, width, channels)` for 2D
    or `(depth, height, width, channels)` for 3D.
- **num_classes (int)**: Number of output classes for classification.
- **classifier_activation (str, optional)**: Activation function for the output layer.
- **decoder_block_type**: Decoder block type, either `upsampling` or `transpose`.
- **decoder_filters**: The projection filters in decoder blocks. Default: `(256, 128, 64, 32, 16)`.
- **name (str)**: Name of the model.

## **DenseUNet201**

A UNet model with a DenseNet-201 backbone

This model is a UNet architecture for image segmentation that uses a DenseNet-201 as its feature-extracting encoder. It's built to provide a powerful and flexible solution for both 2D and 3D segmentation tasks. .

```python
medicai.models.DenseUNet201(
    input_shape,
    num_classes,
    classifier_activation=None,
    decoder_block_type="upsampling",
    decoder_filters=(256, 128, 64, 32, 16),
    name='dense_unet_121',
)

# 3D model.
num_classes = 1
input_shape = (64, 64, 64, 1)
model = medicai.models.DenseUNet201(
    input_shape=input_shape,
    num_classes=num_classes
)

# 2D model.
num_classes = 1
input_shape = (64, 64, 1)
model = medicai.models.DenseUNet201(
    input_shape=input_shape, 
    num_classes=num_classes
)
```

**Reference**

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (CVPR 2015)

**Arguments**

- **input_shape (tuple)**: Shape of the input tensor excluding batch size.
    For example, `(height, width, channels)` for 2D
    or `(depth, height, width, channels)` for 3D.
- **num_classes (int)**: Number of output classes for classification.
- **classifier_activation (str, optional)**: Activation function for the output layer.
- **decoder_block_type**: Decoder block type, either `upsampling` or `transpose`.
- **decoder_filters**: The projection filters in decoder blocks. Default: `(256, 128, 64, 32, 16)`.
- **name (str)**: Name of the model.

## **UNETR**

UNETR: U-Net with a Vision Transformer (ViT) backbone for 3D and 2D medical image segmentation.

UNETR integrates a ViT encoder as the backbone with a UNet-style decoder, using
projection upsampling blocks and skip connections from intermediate transformer layers.


```python
medicai.models.UNETR(
    input_shape,
    num_classes,
    classifier_activation=None,
    feature_size = 16,
    hidden_size = 768,
    mlp_dim = 3072,
    num_heads = 12,
    num_layers = 12,
    patch_size = 16,
    norm_name = "instance",
    conv_block = True,
    res_block = True,
    dropout_rate = 0.0,
    name = "UNETR",
)

# 3D model.
num_classes = 1
input_shape = (64, 64, 64, 1)
model = medicai.models.UNETR(
    input_shape=input_shape, 
    num_classes=num_classes
)

# 2D model.
num_classes = 1
input_shape = (64, 64, 1)
model = medicai.models.UNETR(
    input_shape=input_shape, 
    num_classes=num_classes
)
```

**Reference**

- [UNETR: Transformers for 3D Medical Image Segmentation](https://arxiv.org/abs/2103.10504) (CVPR 2021)

**Arguments**

- **input_shape (tuple)**: Shape of the input tensor excluding batch size.
    For example, `(height, width, channels)` for 2D
    or `(depth, height, width, channels)` for 3D.
- **num_classes (int)**: Number of output segmentation classes.
- **classifier_activation (str, optional)**: Activation function applied to the output layer.
- **feature_size (int)**: Base number of feature channels in decoder blocks.
- **hidden_size (int)**: Hidden size of the transformer encoder.
- **mlp_dim (int)**: Hidden size of MLPs in transformer blocks.
- **num_heads (int)**: Number of attention heads per transformer layer.
- **num_layers (int)**: Number of transformer encoder layers.
- **patch_size (int)**: Size of the patches extracted from input.
- **norm_name (str)**: Type of normalization for decoder blocks (`instance`, `batch`, etc.).
- **conv_block (bool)**: Whether to use convolutional blocks in decoder.
- **res_block (bool)**: Whether to use residual blocks in decoder.
- **dropout_rate (float)**: Dropout rate applied in backbone and intermediate layers.
- **name (str)**: Model name.


## **SwinUNETR**

Swin-UNETR: A hybrid transformer-CNN for 3D or 2D medical image segmentation.

This model combines the strengths of the Swin Transformer for feature extraction
and a U-Net-like architecture for segmentation. It uses a Swin Transformer
backbone to encode the input and a decoder with upsampling and skip connections
to generate segmentation maps.

```python
medicai.models.SwinUNETR(
    input_shape,
    num_classes,
    classifier_activation=None,
    feature_size=48,
    norm_name="instance",
    res_block = True,
    name = "SwinUNETR",
)

# 3D model.
num_classes = 4
input_shape = (96, 96, 96, 1)
model = medicai.models.SwinUNETR(
    input_shape=input_shape, 
    num_classes=num_classes,
    classifier_activation=None
)

# 2D model.
input_shape = (96, 96, 1)
model = medicai.models.SwinUNETR(
    input_shape=input_shape, 
    num_classes=num_classes,
    classifier_activation=None
)
```

**Reference**

- [Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images](https://arxiv.org/abs/2201.01266) (CVPR 2022)

**Arguments**

- **input_shape (tuple)**: Shape of the input tensor excluding batch size.
    For example, `(height, width, channels)` for 2D
    or `(depth, height, width, channels)` for 3D.
- **num_classes (int)**: Number of output segmentation classes.
- **classifier_activation (str, optional)**: Activation function applied to the output layer.
- **feature_size (int)**: The base feature map size in the decoder. Default is `48`.
- **norm_name (str)**: Type of normalization for decoder blocks (`instance`, `batch`, etc.).
- **res_block (bool)**: Whether to use residual blocks in decoder. Default is True.
- **name (str)**: Model name.


## **TransUNet**

TransUNet model for 2D or 3D semantic segmentation.

This model combines a 3D or 2D CNN encoder (DenseNet) with a Vision Transformer
(ViT) encoder and a hybrid decoder. The CNN extracts multi-scale local features,
while the ViT captures global context. The decoder upsamples the fused
features to produce the final segmentation map using a coarse-to-fine
attention mechanism and U-Net-style skip connections.

```python
medicai.models.TransUNet(
    input_shape,
    num_classes,
    patch_size=3,
    classifier_activation=None,
    num_encoder_layers=6,
    num_heads=8,
    num_queries=100,
    embed_dim=256,
    mlp_dim=1024,
    dropout_rate=0.1,
    decoder_projection_filters=64,
    name=None,
)

# 3D model.
num_classes = 4
patch_size = 3
input_shape = (96, 96, 96, 1)
model = medicai.models.TransUNet(
    input_shape=input_shape, 
    num_classes=num_classes,
    patch_size=patch_size,
    classifier_activation=None
)

# 2D model.
input_shape = (96, 96, 1)
model = medicai.models.TransUNet(
    input_shape=input_shape, 
    num_classes=num_classes,
    patch_size=patch_size,
    classifier_activation=None
)
```

**Reference**

- [3D TransUNet: Advancing Medical Image Segmentation through Vision Transformers](https://arxiv.org/abs/2310.07781) (CVPR 2023)

**Arguments**

- **input_shape (tuple)**: The shape of the input data. For 2D, it is
    `(height, width, channels)`. For 3D, it is `(depth, height, width, channels)`.
- **num_classes (int)**: The number of segmentation classes.
- **patch_size (int or tuple)**: The size of the patches for the Vision
    Transformer. Must be a tuple of length `spatial_dims`. Defaults to 3.
- **num_queries (int, optional)**: The number of learnable queries used in the
    decoder's attention mechanism. Defaults to `100`.
- **classifier_activation (str, optional)**: Activation function for the final
    segmentation head (e.g., `sigmoid` for binary, `softmax` for multi-class).
- **num_encoder_layers (int, optional)**: The number of transformer encoder blocks
    in the `ViT` encoder. Defaults to `6`.
- **num_heads (int, optional)**: The number of attention heads in the transformer blocks.
    Defaults to `8`.
- **embed_dim (int, optional)**: The dimensionality of the token embeddings.
    Defaults to `256`.
- **mlp_dim (int, optional)**: The hidden dimension of the MLP in the transformer
    blocks. Defaults to `1024`.
- **dropout_rate (float, optional)**: The dropout rate for regularization.
    Defaults to `0.1`.
- **decoder_projection_filters (int, optional)**: The number of filters for the
    convolutional layers in the decoder upsampling path. Defaults to `64`.
- **name (str, optional)**: The name of the model. Defaults to `TransUNetND`.


**Note**: The **3D-TransUNet** model combines a CNN and a Transformer in its encoder and decoder. While the original version's encoder uses a ResNet-like CNN, the `medicai` implementation uses a Densenet-like feature extractor.


## **SegFormer**

SegFormer model for 2D or 3D semantic segmentation.

This class implements the full SegFormer architecture, which combines a
hierarchical MixVisionTransformer (MiT) encoder with a lightweight MLP decoder
head. This design is highly efficient for semantic segmentation tasks on
high-resolution images or volumes.

The encoder progressively downsamples the spatial dimensions and increases the
feature dimensions across four stages, producing multi-scale feature maps.
The decoder then takes these features, processes them through linear layers,
upsamples them to a common resolution, and fuses them to generate a
high-resolution segmentation mask.

```python
medicai.models.SegFormer(
    input_shape,
    num_classes,
    decoder_head_embedding_dim=256,
    classifier_activation=None,
    qkv_bias=True,
    dropout=0.0,
    project_dim=[32, 64, 160, 256],
    layerwise_sr_ratios=[4, 2, 1, 1],
    layerwise_patch_sizes=[7, 3, 3, 3],
    layerwise_strides=[4, 2, 2, 2],
    layerwise_num_heads=[1, 2, 5, 8],
    layerwise_depths=[2, 2, 2, 2],
    layerwise_mlp_ratios=[4, 4, 4, 4],
    name=None,
)

# 3D model.
num_classes = 4
input_shape = (96, 96, 96, 1)
model = medicai.models.SegFormer(
    input_shape=input_shape, 
    num_classes=num_classes,
)

# 2D model.
input_shape = (96, 96, 1)
model = medicai.models.SegFormer(
    input_shape=input_shape, 
    num_classes=num_classes,
)
```

**Reference**

- [SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation](https://arxiv.org/abs/2404.10156) (CVPR 2024)

**Arguments**

- **input_shape (tuple)**: The shape of the input data, excluding the batch dimension.
- **num_classes (int)**: The number of output classes for segmentation.
- **decoder_head_embedding_dim (int, optional)**: The embedding dimension of the decoder head.
    Defaults to 256.
- **classifier_activation (str, optional)**: The activation function for the final output
    layer. Common choices are `softmax` for multi-class segmentation and `sigmoid` for multi-label or binary segmentation. Defaults to `None`.
- **qkv_bias (bool, optional)**: Whether to include a bias in the query, key, and value    
    projections. Defaults to `True`.
- **dropout (float, optional)**: The dropout rate for the decoder head. Defaults to 0.0.
- **project_dim (list[int], optional)**: A list of feature dimensions for each encoder stage.
    Defaults to `[32, 64, 160, 256]`.
- **layerwise_sr_ratios (list[int], optional)**: A list of spatial reduction ratios for each
    encoder stage's attention layers. Defaults to `[4, 2, 1, 1]`.
- **layerwise_patch_sizes (list[int], optional)**: A list of patch sizes for the embedding
    layer in each encoder stage. Defaults to `[7, 3, 3, 3]`.
- **layerwise_strides (list[int], optional)**: A list of strides for the embedding layer in
    each encoder stage. Defaults to `[4, 2, 2, 2]`.
- **layerwise_num_heads (list[int], optional)**: A list of the number of attention heads for
    each encoder stage. Defaults to `[1, 2, 5, 8]`.
- **layerwise_depths (list[int], optional)**: A list of the number of transformer blocks for
    each encoder stage. Defaults to `[2, 2, 2, 2]`.
- **layerwise_mlp_ratios (list[int], optional)**: A list of MLP expansion ratios for each
    encoder stage. Defaults to `[4, 4, 4, 4]`.
- **name (str, optional)**: The name of the model. Defaults to `None`.