Transformation acts as a prprocessing and augmentation layers for model training. For 3D transformation, the expected format is: `depth, height, width, channel`. The transformation are implemented for a single sample. All the transformations are implemented using `tensorflow` in order to able to run with `tf.data` API with `keras` multi-backend library.

```python
img_path = 'images/coronacases_001.nii.gz'
mask_path = 'masks/coronacases_001.nii.gz'

nib_x = nib.load(img_path) # (512, 512, 301)
nib_y = nib.load(mask_path) # (512, 512, 301)

image = nib_x.get_fdata().transpose(2, 0, 1)[...,None] # (301, 512, 512, 1)
label = nib_y.get_fdata().transpose(2, 0, 1)[...,None] # (301, 512, 512, 1)
```

## Preprocessing

**Resize**

```python
from medicai.transforms import (
    TensorBundle,
    Resize,
)

inputs = TensorBundle({"image": image, "label": label})
transform = Resize(
    keys=["image", "label"], 
    spatial_shape=(96, 96, 96)
)

output = transform(inputs)
transform_image = output.data["image"]
transform_label = output.data["label"]
transform_image.shape, transform_label.shape
(TensorShape([96, 96, 96, 1]), TensorShape([96, 96, 96, 1]))
```

**ScaleIntensityRange**

```python
from medicai.transforms import (
    TensorBundle,
    ScaleIntensityRange,
)

inputs = TensorBundle({"image": image, "label": label})
transform = ScaleIntensityRange(
    keys=["image"], 
    a_min=-175,
    a_max=250,
    b_min=0.0,
    b_max=1.0,
    clip=True
)

output = transform(inputs)
transform_image = output.data["image"]
transform_label = output.data["label"]
transform_image.shape, transform_label.shape

transform_image.numpy().min(), transform_image.numpy().max()
(0.0, 1.0)

np.unique(transform_label)
array([0., 1., 2., 3.])
```

**CropForeground**

```python
from medicai.transforms import (
    CropForeground,
    TensorBundle,
)

inputs = TensorBundle({"image": image, "label": label})
transform = CropForeground(
    keys=("image", "label"), 
    source_key="image"
)
output = transform(inputs)
transform_image = output.data["image"]
transform_label = output.data["label"]
transform_image.shape, transform_label.shape
((301, 512, 415, 1), (301, 512, 415, 1))
```

**Spacing**

```python
from medicai.transforms import (
    TensorBundle,
    Spacing,
)

affine = nib_x.affine
affine[:, [0, 1, 2]] = affine[:, [2, 0, 1]]  # (H, W, D) -> (D, H, W)
trans_affine = affine.astype(np.float32)

inputs = TensorBundle(
    {
        "image": image, 
        "label": label
    }, 
    meta={
        'affine': trans_affine
    }
)
transform = Spacing(
    keys=["image", "label"], 
    pixdim=[2.0, 1.5, 1.5]
)

output = transform(inputs)
transform_image = output.data["image"]
transform_label = output.data["label"]
transform_image.shape, transform_label.shape
(TensorShape([121, 276, 341, 1]), TensorShape([121, 276, 341, 1]))
```

**Orientation**

```python
from medicai.transforms import (
    TensorBundle,
    Orientation,
)

affine = nib_x.affine
affine[:, [0, 1, 2]] = affine[:, [2, 0, 1]]  # (H, W, D) -> (D, H, W)
trans_affine = affine.astype(np.float32)

inputs = TensorBundle(
    {
        "image": image, 
        "label": label
    }, 
    meta={
        'affine': trans_affine
    }
)
transform = Orientation(
    keys=["image", "label"], 
    axcodes="RAS"
)

output = transform(inputs)
transform_image = output.data["image"]
transform_label = output.data["label"]
transform_image.shape, transform_label.shape
(TensorShape([301, 512, 512, 1]), TensorShape([301, 512, 512, 1]))
```

## Radnom Preprocessing

**RandRotate90**

```python
from medicai.transforms import (
    TensorBundle,
    RandRotate90,
)

inputs = TensorBundle({"image": image, "label": label})
transform = RandRotate90(
    keys=["image", "label"], 
    prob=1.0, 
    max_k=3
)
output = transform(inputs)

transform_image = output.data["image"]
transform_label = output.data["label"]
transform_image.shape, transform_label.shape
(TensorShape([301, 512, 512, 1]), TensorShape([301, 512, 512, 1]))
```

**RandShiftIntensity**

```python
from medicai.transforms import (
    TensorBundle,
    RandShiftIntensity,
)

inputs = TensorBundle({"image": image, "label": label})
transform = RandShiftIntensity(
    keys=["image", "label"], 
    offsets=(-0.2, 0.8), 
    prob=1.0
)

output = transform(inputs)
transform_image = output.data["image"]
transform_label = output.data["label"]
transform_image.shape, transform_label.shape
(TensorShape([301, 512, 512, 1]), TensorShape([301, 512, 512, 1]))
```

**RandCropByPosNegLabel**

```python
from medicai.transforms import (
    TensorBundle,
    RandCropByPosNegLabel,
)

inputs = TensorBundle({"image": image, "label": label})
transform = RandCropByPosNegLabel(
    keys=["image", "label"], 
    spatial_size=(96, 96, 96), 
    pos=1, 
    neg=1, 
    num_samples=1
)
output = transform(inputs)
transform_image = output.data["image"]
transform_label = output.data["label"]
transform_image.shape, transform_label.shape
(TensorShape([96, 96, 96, 1]), TensorShape([96, 96, 96, 1]))
```

## Compose

```python
from medicai.transforms import (
    Compose,
    Orientation,
    Spacing,
)

transform = Compose(
    [
        Orientation(keys=["image", "label"], axcodes="RAS"), 
        Spacing(keys=["image", "label"], pixdim=[1.0, 1.2, 1.2])
    ]
)
inputs = {"image": image, "label": label}
meta = {'affine': trans_affine}
output = transform(inputs, meta)

transform_image = output.data["image"]
transform_label = output.data["label"]
transform_image.shape, transform_label.shape
(TensorShape([243, 345, 426, 1]), TensorShape([243, 345, 426, 1]))
```