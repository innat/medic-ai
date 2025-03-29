from medicai.utils.general import hide_warnings

hide_warnings()

from keras import layers


def get_act_layer(act_name):
    if act_name[0] == "leakyrelu":
        return layers.LeakyReLU(negative_slope=act_name[1]["negative_slope"])
    else:
        return layers.Activation(act_name[0])


def get_norm_layer(norm_name):
    if norm_name == "instance":
        return layers.GroupNormalization(groups=-1, epsilon=1e-05, scale=False, center=False)

    elif norm_name == "batch":
        return layers.BatchNormalization()
    else:
        raise ValueError(f"Unsupported normalization: {norm_name}")
