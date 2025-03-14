from keras import applications
from keras import layers

BACKBONE = {
    "efficientnetb0": applications.EfficientNetB0,
    "resnet50": applications.ResNet50,
    "densenet121": applications.DenseNet121,
    "convnextsmall": applications.ConvNeXtSmall,
}

BACKBONE_ARGS = {
    "efficientnetb0": [
        "block6a_expand_activation",
        "block4a_expand_activation",
        "block3a_expand_activation",
        "block2a_expand_activation",
    ],
    "resnet50": ["conv4_block6_2_relu", "conv3_block4_2_relu", "conv2_block3_2_relu", "conv1_relu"],
    "densenet121": [311, 139, 51, 4],
    "convnextsmall": [268, 51, 26],
}


def get_act_layer(act_name):
    if act_name[0] == "leakyrelu":
        return layers.LeakyReLU(negative_slope=act_name[1]["negative_slope"])
    else:
        return layers.Activation(act_name[0])

def get_norm_layer(norm_name):
    if norm_name == "instance":
        return layers.GroupNormalization(
            groups=-1, epsilon=1e-05, scale=False, center=False
        )
        
    elif norm_name == "batch":
        return layers.BatchNormalization()
    else:
        raise ValueError(f"Unsupported normalization: {norm_name}")