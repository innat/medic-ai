from medicai.datasets import APTOSDataloader
from pathlib import Path

db = APTOSDataloader(
    Path("/mnt/c/Users/innat/Desktop/projects/dataset/aptos"),
    subfolder="train_images",
    meta_file="df.csv",
    meta_columns=["id_code", "diagnosis"],
    num_classes=5,
    batch_size=8,
)

x, y = next(iter(db.prepare_sample()))
print(x.shape, y)


x, y = next(iter(db.prepare_samples()))
print(x.shape, y)


from medicai.nets import UNet2D

model = UNet2D(
    backbone='efficientnetb0',
    input_size=224,
    num_class=1,
    class_activation='sigmoid'
)

print(model(x).shape)

print(model.summary())