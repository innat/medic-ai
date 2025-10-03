import keras


class EfficientNet(keras.Model):
    ...

    # if include_top:
    #         x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    #         if dropout_rate > 0:
    #             x = layers.Dropout(dropout_rate, name="top_dropout")(x)
    #         x = layers.Dense(
    #             classes,
    #             activation=classifier_activation,
    #             kernel_initializer="glorot_uniform",
    #             name="predictions",
    #         )(x)
    #     else:
    #         if pooling == "avg":
    #             x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    #         elif pooling == "max":
    #             x = layers.GlobalMaxPooling2D(name="max_pool")(x)
