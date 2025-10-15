import keras

VALID_DECODER_NORMS = (False, "batch", "layer", "unit", "group", "instance", "sync_batch")
VALID_DECODER_BLOCK_TYPE = ("upsampling", "transpose")
VALID_ACTIVATION_LIST = [
    m for m in dir(keras.activations) if not m.startswith("_") and not m[0].isupper()
]
