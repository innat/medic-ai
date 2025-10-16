# SegFormer

**SegFormer** model for 2D or 3D semantic segmentation. It combines a hierarchical **MixVisionTransformer** (MiT) encoder with a lightweight MLP decoder head. This design is highly efficient for semantic segmentation tasks on high-resolution images or volumes.


```python
from medicai.models import SegFormer

# To build 3D model
model = SegFormer(encoder_name='mit_b0', input_shape=(96,96,96,1))

# To build 2D model
model = SegFormer(encoder_name='mit_b0', input_shape=(96,96,1))
```

**Reference**
- [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)
- [SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation](https://arxiv.org/abs/2404.10156)