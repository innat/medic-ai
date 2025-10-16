# TransUNet

This model combines a 3D or 2D CNN encoder with a **Vision Transformer** (ViT) encoder and a hybrid decoder. The CNN extracts multi-scale local features, while the ViT captures global context. The decoder upsamples the fused features to produce the final segmentation map using a **coarse-to-fine** attention mechanism and U-Net-style skip connections.


```python
from medicai.models import TransUNet

# To build 3D model
model = TransUNet(encoder_name='resnet18', input_shape=(96,96,96,1))

# To build 2D model
model = TransUNet(encoder_name='resnet18', input_shape=(96,96,1))
```

**Reference**
- [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306)
- [3D TransUNet: Advancing Medical Image Segmentation through Vision Transformers](https://arxiv.org/abs/2310.07781)