# TransUNet

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

---

**Note**: There are two publised TransUNet for 2D and 3D, with different model architecture. The 3D TransUNet is much more complex than 2D TransUNet. Also, as mentioned in the 3D TransUNet paper, this model can also be used for 2D task by replacing 3D ops to 2D ops. In this codebase, only 3D TransUNet is implemented. If the 2D input shape is passed, the built model would be basically 3D TransUNet version in 2D. 

**Reference**
- [2D TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306)
- [3D TransUNet: Advancing Medical Image Segmentation through Vision Transformers](https://arxiv.org/abs/2310.07781)