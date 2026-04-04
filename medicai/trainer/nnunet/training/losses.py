"""
medicai/trainer/nnunet/training/losses.py
======================================
Loss functions for nnU-Net training.
Note: Deep supervision is now handled via multi-output dictionaries
and standard Keras 3 loss weighting in the model/trainer.
"""

import keras
from keras import ops

# Custom nnU-Net specific loss extensions can be added here.
