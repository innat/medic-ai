from abc import ABC, abstractmethod

import keras
from keras import ops

from medicai.layers import ResizingND


class BaseCAM(ABC):
    def __init__(self, model, target_layer, task_type="auto"):
        self.model = model
        self.target_layer = target_layer
        self.task_type = task_type
        self.backend = keras.config.backend()
        print(f"Using backend: {self.backend}")

        # Auto-detect task type
        if self.task_type == "auto":
            self.detect_task_type()

        # Build grad model
        self.grad_model = self.build_grad_model()

    def detect_task_type(self):
        output_shape = self.model.output.shape
        if len(output_shape) == 2:  # (B, num_classes) - Classification
            self.task_type = "classification"
        elif len(output_shape) in (4, 5):  # (B, [D], H, W, C) - Segmentation
            self.task_type = "segmentation"
        else:
            raise ValueError(f"Unknown output shape: {output_shape}")

        print(f"Auto-detected task type: {self.task_type}")

    def build_grad_model(self):
        try:
            target_layer_obj = self.model.get_layer(self.target_layer)
        except ValueError:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model")

        if self.backend == "tensorflow":
            grad_model = keras.Model(
                inputs=self.model.input, outputs=[target_layer_obj.output, self.model.output]
            )
            return grad_model
        else:
            hidden_model = keras.Model(inputs=self.model.inputs, outputs=target_layer_obj.output)
            # Layers after target layer → classifier model
            target_idx = self.model.layers.index(target_layer_obj)
            classifier_input = keras.Input(shape=target_layer_obj.output.shape[1:])
            x = classifier_input
            for layer in self.model.layers[target_idx + 1 :]:
                x = layer(x)
            classifier_model = keras.Model(classifier_input, x)
            return classifier_model, hidden_model

    def compute_target(self, predictions, target_class_index, mask_type):
        if self.task_type == "classification":
            # Classification: use single class score
            if target_class_index is None:
                target_class_index = ops.argmax(predictions[0])
            target = predictions[0, target_class_index]  # Single scalar
            return target

        else:  # Segmentation (2D or 3D)
            num_classes = predictions.shape[-1]

            if num_classes == 1:  # Binary
                y_c_ij = ops.squeeze(predictions, axis=-1)
                if mask_type == "object":
                    M = ops.cast(y_c_ij > 0.0, "float32")
                elif mask_type == "all":
                    M = ops.ones_like(y_c_ij)
                elif mask_type == "single":
                    max_val = ops.max(y_c_ij)
                    M = ops.cast(ops.equal(y_c_ij, max_val), "float32")
                target = ops.sum(y_c_ij * M)

            else:  # Multi-class
                if target_class_index is None:
                    raise ValueError("target_class_index required for multi-class segmentation")

                y_c_ij = predictions[..., target_class_index]
                predicted_classes = ops.argmax(predictions, axis=-1)

                if mask_type == "object":
                    M = ops.cast(ops.equal(predicted_classes, target_class_index), "float32")
                elif mask_type == "all":
                    M = ops.ones_like(y_c_ij)
                elif mask_type == "single":
                    flat_y = ops.reshape(y_c_ij, [ops.shape(y_c_ij)[0], -1])
                    max_val = ops.max(flat_y, axis=-1, keepdims=True)
                    M_flat = ops.cast(ops.equal(flat_y, max_val), "float32")
                    M = ops.reshape(M_flat, ops.shape(y_c_ij))

                # Debug information
                print(f"y_c_ij range: [{ops.min(y_c_ij):.3f}, {ops.max(y_c_ij):.3f}]")
                print(f"M sum: {ops.sum(M)}")
                print(f"Target: {ops.sum(y_c_ij * M):.3f}")
                target = ops.sum(y_c_ij * M)

            return target

    def resize_heatmap(self, heatmap, target_shape):
        # Add channel dimension if needed
        if len(heatmap.shape) == len(target_shape):
            heatmap = ops.expand_dims(heatmap, axis=-1)
        resized_layer = ResizingND(
            target_shape, interpolation="bilinear" if len(target_shape) == 2 else "trilinear"
        )
        return resized_layer(heatmap)

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
        except ImportError:
            raise ImportError("TensorFlow is not installed. Please install tensorflow.")

        with tf.GradientTape() as tape:
            hidden_output, predictions = self.grad_model(input_tensor)
            tape.watch(hidden_output)
            target = self.compute_target(predictions, target_class_index, mask_type)
        grads = tape.gradient(target, hidden_output)
        return grads, hidden_output, predictions

    def _compute_gradients_jax(self, input_tensor, target_class_index, mask_type):
        try:
            import jax
        except ImportError:
            raise ImportError("JAX is not installed. Please install jax and jaxlib.")

        classifier_model, hidden_model = self.grad_model

        def loss_fn(hidden_output):
            predictions = classifier_model(hidden_output)
            target = self.compute_target(predictions, target_class_index, mask_type)
            return target, predictions

        # Gradient of the loss wrt convolutional output
        value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        # Compute hidden activations and gradients
        hidden_output = hidden_model(input_tensor)
        (target_value, predictions), grads = value_and_grad_fn(hidden_output)
        return grads, hidden_output, predictions

    def _compute_gradients_torch(self, input_tensor, target_class_index, mask_type):
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is not installed.")

        classifier_model, hidden_model = self.grad_model
        hidden_output = hidden_model(input_tensor)
        hidden_output = hidden_output.clone().detach().requires_grad_(True)
        predictions = classifier_model(hidden_output)

        # Get the specific class score
        target = self.compute_target(predictions, target_class_index, mask_type)

        # Backward pass to compute gradients
        target.backward()

        # Get gradients
        grads = hidden_output.grad

        return grads, hidden_output, predictions

    @abstractmethod
    def compute_heatmap(self, input_tensor, target_class_index=None, mask_type="object"):
        """Compute CAM heatmap - to be implemented by subclasses"""
        pass


class GradCAM(BaseCAM):
    def compute_heatmap(self, input_tensor, target_class_index=None, mask_type="object"):

        # Compute gradients using unified backend method
        grads, hidden_output, predictions = self.compute_gradients(
            input_tensor, target_class_index, mask_type
        )

        if grads is None:
            raise ValueError("Gradient computation failed - no gradients returned")

        # Global average pooling of gradients
        spatial_axes = tuple(range(1, len(grads.shape) - 1))
        pooled_grads = ops.mean(grads, axis=spatial_axes)
        pooled_grads = ops.reshape(
            pooled_grads, [1] * (len(hidden_output.shape) - 1) + [pooled_grads.shape[-1]]
        )

        # Get feature maps for this image (remove batch dimension)
        hidden_output = hidden_output[0]

        # Weighted combination of feature maps
        heatmap = ops.sum(hidden_output * pooled_grads, axis=-1)

        # Apply ReLU
        heatmap = ops.relu(heatmap)

        # Normalize
        heatmap_max = ops.max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / (heatmap_max + keras.backend.epsilon())
        else:
            heatmap = ops.zeros_like(heatmap)

        # Resize to original spatial dimensions
        heatmap = heatmap[..., None]
        heatmap = self.resize_heatmap(heatmap, tuple(input_tensor.shape[1:-1]))

        return heatmap
