from tensorflow.keras import applications

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
