# TransUNet

![](https://github.com/user-attachments/assets/a9f568d8-5e17-433f-a71d-4ed5c94da280)

This model combines a 3D or 2D CNN encoder with a **Vision Transformer** (ViT) encoder and a hybrid decoder. The CNN extracts multi-scale local features, while the ViT captures global context. The decoder upsamples the fused features to produce the final segmentation map using a **coarse-to-fine** attention mechanism and U-Net-style skip connections.


```python
from medicai.models import TransUNet

# To build 3D model (with resnet)
model = TransUNet(encoder_name='resnet18', input_shape=(96,96,96,1))

# To build 2D model (with efficientnet)
model = TransUNet(encoder_name='efficientnet_b2', input_shape=(96,96,1))

# To reduce the size of the model (with densenet)
model = TransUNet(
    encoder_name='densenet121', 
    input_shape=(96,96,96,1),
    encoder_depth=3, # 3 or 4, default (5)
    num_vit_layers=6, # default (12)
    num_heads=8,
    embed_dim=512,
    mlp_dim=1024,
)
```

**Note**: There are two published **TransUNet** for 2D and 3D task, with different decoder modelling. In 3D **TransUNet**, it is mentioned that, this model can be applied in 2D task by replacing the 3D ops to 2D ops. In this codebase, 3D version of TransUNet is implemented. If 2D input shape is passed, the built model would be 3D TransUNet in 2D version.

**Reference**
- [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306)
- [3D TransUNet: Advancing Medical Image Segmentation through Vision Transformers](https://arxiv.org/abs/2310.07781)