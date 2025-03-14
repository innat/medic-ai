
from keras import layers
from medicai.utils import get_act_layer, get_norm_layer

def UnetResBlock(in_channels, out_channels, kernel_size=3, stride=1, norm_name="instance", dropout_rate=None):
    def wrapper(inputs):
        # first convolution
        x = layers.Conv3D(out_channels, kernel_size, strides=stride, padding='same', use_bias=False)(inputs)
        x = get_norm_layer(norm_name)(x)
        x = get_act_layer(("leakyrelu", {"inplace": True, "negative_slope": 0.01}))(x)
        
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)
        
        # second convolution
        x = layers.Conv3D(out_channels, kernel_size, strides=1, padding='same', use_bias=False)(x)
        x = get_norm_layer(norm_name)(x)
        
        # residual
        residual = inputs
        downsample = (in_channels != out_channels) or (np.atleast_1d(stride) != 1).any()
        if downsample:
            residual = layers.Conv3D(
                out_channels, kernel_size=1, strides=stride, padding='same', use_bias=False
            )(residual)
            residual = get_norm_layer(norm_name)(residual)
        
        # add residual connection
        x = layers.Add()([x, residual])
        x = get_act_layer(("leakyrelu", {"inplace": True, "negative_slope": 0.01}))(x)
        
        return x
    return wrapper