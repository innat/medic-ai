
import keras
from keras import ops

class DiceCELoss(keras.losses.Loss):
    def __init__(self, to_onehot_y=True, softmax=True, smooth=1e-5):
        super(DiceCELoss, self).__init__()
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax
        self.smooth = smooth

    def dice_loss(self, y_true, y_pred):
        intersection = ops.sum(y_true * y_pred, axis=[1, 2, 3])
        union = ops.sum(y_true, axis=[1, 2, 3]) + ops.sum(y_pred, axis=[1, 2, 3])
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - ops.mean(dice_score)

    def call(self, y_true, y_pred):
        if self.to_onehot_y:
            num_classes = ops.shape(y_pred)[-1]
            y_true = ops.squeeze(y_true, axis=-1)
            y_true = ops.one_hot(ops.cast(y_true, tf.int32), num_classes=num_classes)
        
        if self.softmax:
            y_pred = ops.nn.softmax(y_pred, axis=-1)

        dice = self.dice_loss(y_true, y_pred)
        ce = keras.losses.categorical_crossentropy(y_true, y_pred)
        return dice + ops.mean(ce)