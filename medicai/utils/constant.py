import keras

VALID_DECODER_NORMS = (False, "batch", "layer", "unit", "group", "instance", "sync_batch")
VALID_DECODER_BLOCK_TYPE = ("upsampling", "transpose")

_ACTIVATION_EXCLUDES = {"get", "serialize", "deserialize"}
VALID_ACTIVATION_LIST = [
    m
    for m in dir(keras.activations)
    if (not m.startswith("_") and not m[0].isupper() and m not in _ACTIVATION_EXCLUDES)
]
