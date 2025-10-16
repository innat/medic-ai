# UNETR

**UNETR**, a U-Net with a Vision Transformer (ViT) backbone for 3D and 2D medical image segmentation.

**UNETR** integrates a ViT encoder as the backbone with a UNet-style decoder, using
projection upsampling blocks and skip connections from intermediate transformer layers.
It is designed to leverage the global context-modeling power of Transformers for
high-resolution tasks like medical image segmentation.


```python
from medicai.models import UNETR

# To build 3D model
model = UNETR(encoder_name='vit_base', input_shape=(96,96,96,1))
model.count_params() / 1e6
# 92.811825

# To build 2D model
model = UNETR(encoder_name='vit_base', input_shape=(96,96,1))
model.count_params() / 1e6
# 87.174417
```

**Reference**
- [UNETR: Transformers for 3D Medical Image Segmentation](https://arxiv.org/abs/2103.10504)
