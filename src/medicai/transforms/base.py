from src.medicai.transforms.tensor_bundle import MetaTensor


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image_data, meta_data=None):
        x = MetaTensor(image_data, meta_data)
        for transform in self.transforms:
            x = transform(x)
        return x
