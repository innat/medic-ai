import logging
from abc import ABC, abstractmethod

import keras
from keras import ops

from .general import BaseEnum
from .image import resize_volumes

logger = logging.getLogger(__name__)


class TaskType(BaseEnum):
    AUTO = "auto"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"


class MaskType(BaseEnum):
    OBJECT = "object"
    ALL = "all"
    SINGLE = "single"


class BaseCAM(ABC):
    def __init__(self, model, target_layer, task_type: TaskType = TaskType.AUTO):
        self.model = model
        self.target_layer = target_layer

        self.backend = keras.config.backend()
        logger.info(f"Using backend: {self.backend}")

        task_type = task_type.value if isinstance(task_type, TaskType) else str(task_type)
        if task_type not in TaskType.values():
            raise ValueError(
                f"Invalid task_type: '{task_type}'. Must be one of {TaskType.values()}"
            )
        self.task_type = task_type

        # Auto-detect task type
        if self.task_type == TaskType.AUTO.value:
            self.detect_task_type()

        # Build grad model
        self.grad_model = self.build_grad_model()

    def detect_task_type(self):
        output_shape = self.model.output.shape
        rank = len(output_shape)

        if rank == 2:
            # (Batch, Channel)
            self.task_type = TaskType.CLASSIFICATION.value
        elif rank == 4 or rank == 5:
            # (Batch, [Depth], Height, Width, Channel)
            self.task_type = TaskType.SEGMENTATION.value
        else:
            raise ValueError(f"Unknown output shape: {output_shape}")

        logger.info(f"Auto-detected task type: {self.task_type}")

    def build_grad_model(self):
        try:
            target_layer_obj = self.model.get_layer(self.target_layer)
        except ValueError as err:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model") from err

        if self.backend == "tensorflow":
            grad_model = keras.Model(
                inputs=self.model.inputs, outputs=[target_layer_obj.output, self.model.output]
            )
            return grad_model
        elif self.backend == "torch":
            return self.model
        elif self.backend == "jax":
            hidden_model = keras.Model(self.model.inputs, target_layer_obj.output)
            classifier_model = keras.Model(target_layer_obj.output, self.model.output)
            return classifier_model, hidden_model
        else:
            raise ValueError(f"Unsupported backend for GradCAM: {self.backend}")

    def compute_target(self, predictions, target_class_index, mask_type):
        batch_size = ops.shape(predictions)[0]
        if self.task_type == "classification":
            # Classification: use single class score
            if target_class_index is None:
                target_indices = ops.argmax(predictions, axis=-1)
            else:
                target_indices = ops.ones(batch_size, dtype="int32") * target_class_index

            target = ops.take_along_axis(
                predictions, ops.expand_dims(target_indices, axis=-1), axis=-1
            )
            target = ops.squeeze(target, axis=-1)
            return target

        else:  # Segmentation (2D or 3D)
            num_classes = predictions.shape[-1]

            if num_classes == 1:  # Binary
                y_c_ij = ops.squeeze(predictions, axis=-1)
                class_mask = self._get_segmentation_mask(mask_type, y_c_ij)
                target = ops.sum(y_c_ij * class_mask, axis=tuple(range(1, y_c_ij.ndim)))

            else:  # Multi-class
                if target_class_index is None:
                    raise ValueError("target_class_index required for multi-class segmentation")

                y_c_ij = predictions[..., target_class_index]
                predicted_classes = ops.argmax(predictions, axis=-1)
                class_mask = self._get_segmentation_mask(
                    mask_type, y_c_ij, predicted_classes, target_class_index
                )
                target = ops.sum(y_c_ij * class_mask, axis=tuple(range(1, y_c_ij.ndim)))

            return target

    def _get_segmentation_mask(
        self, mask_type, y_c_ij, predicted_classes=None, target_class_index=None
    ):
        if mask_type == "all":
            return ops.ones_like(y_c_ij)
        if mask_type == "single":
            max_val = ops.max(y_c_ij, axis=tuple(range(1, y_c_ij.ndim)), keepdims=True)
            return ops.cast(ops.equal(y_c_ij, max_val), "float32")
        if mask_type == "object":
            if predicted_classes is not None:  # Multi-class
                return ops.cast(ops.equal(predicted_classes, target_class_index), "float32")
            else:  # Binary
                return ops.cast(y_c_ij > 0.0, "float32")

    def resize_heatmap(self, heatmap, target_shape):
        if len(target_shape) == 2:
            resized = ops.image.resize(heatmap, target_shape, interpolation="bilinear")
        elif len(target_shape) == 3:
            resized = resize_volumes(heatmap, *target_shape, method="trilinear")
        else:
            raise ValueError(f"Unsupported target_shape: {target_shape}")
        return resized

    def compute_gradients(self, input_tensor, target_class_index=None, mask_type="object"):
        if self.backend == "tensorflow":
            return self._compute_gradients_tf(input_tensor, target_class_index, mask_type)
        elif self.backend == "jax":
            return self._compute_gradients_jax(input_tensor, target_class_index, mask_type)
        elif self.backend == "torch":
            return self._compute_gradients_torch(input_tensor, target_class_index, mask_type)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _compute_gradients_tf(self, input_tensor, target_class_index, mask_type):
        try:
            import tensorflow as tf
        except ImportError as err:
            raise ImportError("TensorFlow is not installed. Please install tensorflow.") from err

        with tf.GradientTape() as tape:
            hidden_output, predictions = self.grad_model(input_tensor)
            tape.watch(hidden_output)
            target = self.compute_target(predictions, target_class_index, mask_type)
        grads = tape.gradient(target, hidden_output)
        return grads, hidden_output, predictions

    def _compute_gradients_jax(self, input_tensor, target_class_index, mask_type):
        try:
            import jax
        except ImportError as err:
            raise ImportError("JAX is not installed. Please install jax and jaxlib.") from err

        classifier_model, hidden_model = self.grad_model

        def loss_fn(hidden_output):
            predictions = classifier_model(hidden_output)
            target = self.compute_target(predictions, target_class_index, mask_type)
            target = ops.sum(target)
            return target, predictions

        # Gradient of the loss wrt convolutional output
        value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        # Compute hidden activations and gradients
        hidden_output = hidden_model(input_tensor)
        (_, predictions), grads = value_and_grad_fn(hidden_output)
        return grads, hidden_output, predictions

    def _compute_gradients_torch(self, input_tensor, target_class_index, mask_type):
        try:
            import torch
        except ImportError as err:
            raise ImportError("PyTorch is not installed.") from err

        # evaluation mode
        self.model.eval()
        target_layer_obj = self.model.get_layer(self.target_layer)

        activations = []
        gradients = []

        # Forward hook to capture activations
        def forward_hook(module, input, output):
            activations.append(output)

        # Backward hook to capture gradients
        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        # Register hooks
        handle_fwd = target_layer_obj.register_forward_hook(forward_hook)
        handle_bwd = target_layer_obj.register_backward_hook(backward_hook)

        try:
            preds = self.model(input_tensor)
            target = self.compute_target(preds, target_class_index, mask_type)
            self.model.zero_grad()
            target.backward(gradient=torch.ones_like(target))
            hidden_output = activations[0]
            grads = gradients[0]
        finally:
            # Remove hooks
            handle_fwd.remove()
            handle_bwd.remove()

        return grads, hidden_output, preds

    @abstractmethod
    def compute_heatmap(self, input_tensor, target_class_index=None, mask_type="object"):
        """Compute CAM heatmap - to be implemented by subclasses"""
        pass


class GradCAM(BaseCAM):
    def compute_heatmap(self, input_tensor, target_class_index=None, mask_type="object"):

        mask_type = mask_type.value if isinstance(mask_type, MaskType) else str(mask_type)
        if mask_type not in MaskType.values():
            raise ValueError(
                f"Unsupported mask_type '{mask_type}'. Use one of: {MaskType.values()}"
            )

        # Compute gradients using unified backend method
        grads, hidden_output, _ = self.compute_gradients(
            input_tensor, target_class_index, mask_type
        )

        if grads is None:
            raise ValueError("Gradient computation failed - no gradients returned")

        # Global average pooling of gradients
        spatial_axes = tuple(range(1, len(grads.shape) - 1))
        pooled_grads = ops.mean(grads, axis=spatial_axes, keepdims=True)

        # Weighted combination of feature maps
        heatmap = ops.sum(hidden_output * pooled_grads, axis=-1)

        # Apply ReLU
        heatmap = ops.relu(heatmap)

        # Normalize
        heatmap_max = ops.max(heatmap, axis=tuple(range(1, len(heatmap.shape))), keepdims=True)
        eps = keras.config.epsilon()
        heatmap = ops.where(heatmap_max > 0, heatmap / (heatmap_max + eps), ops.zeros_like(heatmap))

        # Resize to original spatial dimensions
        heatmap = ops.expand_dims(heatmap, axis=-1)
        heatmap = self.resize_heatmap(heatmap, tuple(input_tensor.shape[1:-1]))
        heatmap = ops.convert_to_numpy(heatmap)

        return heatmap
