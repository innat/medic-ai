# TransUNet

![](https://github.com/user-attachments/assets/a9f568d8-5e17-433f-a71d-4ed5c94da280)

This model combines a 3D or 2D CNN encoder with a **Vision Transformer** (ViT) encoder and a hybrid decoder. The CNN extracts multi-scale local features, while the ViT captures global context. The decoder upsamples the fused features to produce the final segmentation map using a **coarse-to-fine** attention mechanism and U-Net-style skip connections.


```python
from medicai.models import TransUNet

# To build 3D model
model = TransUNet(encoder_name='resnet18', input_shape=(96,96,96,1))

# To build 2D model
model = TransUNet(encoder_name='resnet18', input_shape=(96,96,1))
```

---

**Note**: For the 2D version of the **TransUNet** model, we adopted the decoder architecture from the 3D **TransUNet**. The 3D variant demonstrates significantly greater modeling capability compared to the 2D version. The 2D version of the 3D model is also mentioned in the 3D-TransUNet paper. From the paper, page 4:

> Lastly, we would like to note that our method, though built upon the 3D nnU-Net, can be easily modified to fit 2D tasks by simply switching the backbone model and reducing all
operations back to 2D.


**Reference**
- [3D TransUNet: Advancing Medical Image Segmentation through Vision Transformers](https://arxiv.org/abs/2310.07781)