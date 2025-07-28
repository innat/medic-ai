from keras import layers


def get_layer_fn(ndim):
    if ndim == 3:
        return {
            "Conv": layers.Conv2D,
            "AvgPool": layers.AveragePooling2D,
            "MaxPool": layers.MaxPooling2D,
            "GlobalAvgPool": layers.GlobalAveragePooling2D,
            "GlobalMaxPool": layers.GlobalMaxPooling2D,
            "BN": layers.BatchNormalization,
        }
    elif ndim == 4:
        return {
            "Conv": layers.Conv3D,
            "AvgPool": layers.AveragePooling3D,
            "MaxPool": layers.MaxPooling3D,
            "GlobalAvgPool": layers.GlobalAveragePooling3D,
            "GlobalMaxPool": layers.GlobalMaxPooling3D,
            "BN": layers.BatchNormalization,
        }
    else:
        raise ValueError(f"Unsupported ndim={ndim}")
