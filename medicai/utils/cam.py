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
    """
    Base class for Class Activation Map (CAM) methods.

    This abstract class handles the fundamental setup for CAM visualizations,
    including model initialization, target layer identification, backend detection,
    task type auto-detection, and unified gradient computation across different Keras backends.

    Subclasses must implement the `compute_heatmap` method.
    """

    def __init__(self, model, target_layer, task_type: TaskType = TaskType.AUTO):
        """
        Initializes the BaseCAM object.

        Args:
            model: The Keras model to analyze.
            target_layer: The name of the convolutional layer whose feature maps will be used.
            task_type: Specifies the type of task. Can be a `TaskType` enum member
                (e.g., `TaskType.CLASSIFICATION`) or its string equivalent
                (e.g., `'classification'`). If 'auto', it is inferred from the
                model's output shape.

        Raises:
            ValueError: If an invalid task_type is provided.
        """
        self.model = model
        self.target_layer = target_layer

        # Detect the current Keras backend (tensorflow, torch, jax)
        self.backend = keras.config.backend()
        logger.info(f"Using backend: {self.backend}")

        # Validate and set task type
        task_type = task_type.value if isinstance(task_type, TaskType) else str(task_type)
        if task_type not in TaskType.values():
            raise ValueError(
                f"Invalid task_type: '{task_type}'. Must be one of {TaskType.values()}"
            )
        self.task_type = task_type

        # Auto-detect task type if set to AUTO
        if self.task_type == TaskType.AUTO.value:
            self.detect_task_type()

        # Build the model required for gradient computation, which varies by backend
        self.grad_model = self.build_grad_model()

    def detect_task_type(self):
        """
        Infers the model's task type (CLASSIFICATION or SEGMENTATION) based on the
        rank of the model's output shape.

        - Rank 2: (Batch, Channel) -> Classification
        - Rank 4 or 5: (Batch, [Depth], Height, Width, Channel) -> Segmentation

        Raises:
            ValueError: If the output shape rank is not recognized.
        """
        output_shape = self.model.output.shape
        rank = len(output_shape)

        # Determine task based on output rank
        if rank == 2:
            # Output is (Batch, Classes)
            self.task_type = TaskType.CLASSIFICATION.value
        elif rank == 4 or rank == 5:
            # Output is (Batch, H, W, Classes) or (Batch, D, H, W, Classes)
            self.task_type = TaskType.SEGMENTATION.value
        else:
            raise ValueError(f"Unknown output shape: {output_shape}")

        logger.info(f"Auto-detected task type: {self.task_type}")

    def build_grad_model(self):
        """
        Constructs the specific model(s) needed for gradient computation, which
        differs based on the Keras backend (TF, Torch, JAX).

        Returns:
            A Keras Model or a tuple of Models, depending on the backend, designed
            to extract both the target layer's feature maps and the final prediction.

        Raises:
            ValueError: If the target layer is not found in the model.
            ValueError: If the Keras backend is unsupported.
        """

        try:
            target_layer_obj = self.model.get_layer(self.target_layer)
        except ValueError as err:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model") from err

        if self.backend == "tensorflow":
            # TF: Create a single model that outputs both
            # the target layer features and final predictions.
            grad_model = keras.Model(
                inputs=self.model.inputs, outputs=[target_layer_obj.output, self.model.output]
            )
            return grad_model
        elif self.backend == "torch":
            # PyTorch: Gradient computation uses forward/backward hooks on the original model.
            # We return the original model for simplicity in the compute_gradients dispatcher.
            return self.model
        elif self.backend == "jax":
            # JAX: We need two separate models: one for the feature extraction and one for the
            # classifier head. This allows jax.value_and_grad to be applied correctly.
            hidden_model = keras.Model(self.model.inputs, target_layer_obj.output)
            classifier_model = keras.Model(target_layer_obj.output, self.model.output)
            return classifier_model, hidden_model
        else:
            raise ValueError(f"Unsupported backend for GradCAM: {self.backend}")

    def compute_target(self, predictions, target_class_index, mask_type):
        """
        Calculates the scalar score or tensor to differentiate against.

        This target value represents the model's confidence in the area/class of interest.

        Args:
            predictions: The output tensor from the model.
            target_class_index: The index of the class to focus on. If None, the
                                highest predicted class is used for classification.
            mask_type: For segmentation, specifies how to mask the prediction
                ('all', 'single', 'object').

        Returns:
            A 1D tensor of scalar target values for the batch.
        """
        batch_size = ops.shape(predictions)[0]
        if self.task_type == "classification":
            # Classification: Target is the score of the selected class (scalar per batch item).
            if target_class_index is None:
                # Use the index of the highest prediction
                target_indices = ops.argmax(predictions, axis=-1)
            else:
                # Use the user-specified index
                target_indices = ops.ones(batch_size, dtype="int32") * target_class_index

            # Select the probability/logit for the target class for each sample
            target = ops.take_along_axis(
                predictions, ops.expand_dims(target_indices, axis=-1), axis=-1
            )
            # Remove the last axis, resulting in (Batch,) shape
            target = ops.squeeze(target, axis=-1)
            return target

        else:  # Segmentation (2D or 3D)
            num_classes = predictions.shape[-1]

            if num_classes == 1:  # Binary
                y_c_ij = ops.squeeze(predictions, axis=-1)
                # Compute a spatial mask
                class_mask = self._get_segmentation_mask(mask_type, y_c_ij)
                # Target is the sum of masked scores over all spatial dimensions
                target = ops.sum(y_c_ij * class_mask, axis=tuple(range(1, y_c_ij.ndim)))

            else:  # Multi-class
                if target_class_index is None:
                    raise ValueError("target_class_index required for multi-class segmentation")

                # Extract score for the specific target class
                y_c_ij = predictions[..., target_class_index]

                # Determine the predicted class at every pixel/voxel
                predicted_classes = ops.argmax(predictions, axis=-1)

                # Compute the spatial mask for the target class
                class_mask = self._get_segmentation_mask(
                    mask_type, y_c_ij, predicted_classes, target_class_index
                )

                # Target is the sum of masked scores over all spatial dimensions
                target = ops.sum(y_c_ij * class_mask, axis=tuple(range(1, y_c_ij.ndim)))

            return target

    def _get_segmentation_mask(
        self, mask_type, y_c_ij, predicted_classes=None, target_class_index=None
    ):
        """
        Generates a spatial mask used for segmentation target calculation.

        Args:
            mask_type: The type of mask to generate ('all', 'single', 'object').
            y_c_ij: The prediction tensor for the target class (spatial dimensions only).
            predicted_classes: The class index predicted at each spatial location (multi-class only).
            target_class_index: The index of the class being analyzed (multi-class only).

        Returns:
            A tensor mask of the same shape as y_c_ij, with float32 values (0.0 or 1.0).
        """
        if mask_type == "all":
            # Mask all spatial locations
            return ops.ones_like(y_c_ij)
        if mask_type == "single":
            # Mask only the location(s) with the maximum prediction value for the target class
            max_val = ops.max(y_c_ij, axis=tuple(range(1, y_c_ij.ndim)), keepdims=True)
            return ops.cast(ops.equal(y_c_ij, max_val), "float32")
        if mask_type == "object":
            if predicted_classes is not None:  # Multi-class: use true object mask
                # Mask where the predicted class matches the target class
                return ops.cast(ops.equal(predicted_classes, target_class_index), "float32")
            else:  # Binary
                # Mask where the prediction score is above 0.0
                return ops.cast(y_c_ij > 0.0, "float32")

    def resize_heatmap(self, heatmap, target_shape):
        """
        Resizes the computed heatmap to match the original input spatial dimensions.

        Args:
            heatmap: The computed heatmap tensor.
            target_shape: The (H, W) or (D, H, W) dimensions of the original input.

        Returns:
            The resized heatmap tensor.
        """
        if len(target_shape) == 2:
            resized = ops.image.resize(heatmap, target_shape, interpolation="bilinear")
        elif len(target_shape) == 3:
            resized = resize_volumes(heatmap, *target_shape, method="trilinear")
        else:
            raise ValueError(f"Unsupported target_shape: {target_shape}")
        return resized

    def compute_gradients(self, input_tensor, target_class_index=None, mask_type="object"):
        """
        Computes the gradient of the target output score with respect to the
        target layer's feature maps, abstracting the backend differences.

        Args:
            input_tensor: The input image/volume tensor.
            target_class_index: The class index to focus on (see compute_target).
            mask_type: The mask type for segmentation (see compute_target).

        Returns:
            A tuple: (grads, hidden_output, predictions).

        Raises:
            ValueError: If the backend is unsupported.
        """
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
        training_flag = self.grad_model.training
        self.grad_model.eval()
        target_layer_obj = self.grad_model.get_layer(self.target_layer)

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
            preds = self.grad_model(input_tensor)
            target = self.compute_target(preds, target_class_index, mask_type)
            self.grad_model.zero_grad()
            target.backward(gradient=torch.ones_like(target))
            hidden_output = activations[0]
            grads = gradients[0]
        finally:
            # Remove hooks
            handle_fwd.remove()
            handle_bwd.remove()
            if training_flag:
                self.grad_model.train()

        return grads, hidden_output, preds

    @abstractmethod
    def compute_heatmap(self, input_tensor, target_class_index=None, mask_type="object"):
        """Compute CAM heatmap - to be implemented by subclasses"""
        pass


class GradCAM(BaseCAM):
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM). Grad-CAM generates coarse
    localization heatmaps highlighting the spatial regions that most strongly influence
    a model’s prediction for a target class. The method works by:

    1. Computing gradients of the target prediction with respect to feature maps
       from a selected convolutional layer.
    2. Global-average-pooling the gradients to obtain channel importance weights.
    3. Computing a weighted combination of feature maps.
    4. Applying ReLU to retain only positive contributions.

    This implementation supports:

    - 2D image inputs
    - 3D volumetric inputs
    - Classification models
    - Segmentation models
    - TensorFlow, JAX, and PyTorch Keras backends

    For segmentation models, different masking strategies are supported to
    control how target regions contribute to gradient computation.

    Args:
        model: The Keras model to analyze.
        target_layer: The name of the convolutional layer whose feature maps will be used.
        task_type: Specifies the type of task. Can be "auto", "classification", or "segmentation". 
            If "auto", it is inferred from the model's output shape.

    Raises:
        ValueError: If an invalid task_type is provided.

    .. note::
        - For segmentation models with more than one output channel,
          ``target_class_index`` must be provided when calling
          :meth:`compute_heatmap`.
        - The ``object`` mask for multi-channel segmentation uses
          ``argmax(predictions, axis=-1)`` to determine the predicted region for
          the requested class. This works best for mutually exclusive classes.
          For overlapping target channels, such as region-wise BraTS labels,
          ``mask_type="all"`` is often a better choice.
        - For very large 3D volumes, Grad-CAM is typically applied patch-wise in
          a notebook or user pipeline, then reconstructed with
          ``medicai.utils.extract_patches`` and ``medicai.utils.merge_patches``.
    """

    def compute_heatmap(
        self,
        input_tensor,
        target_class_index=None,
        mask_type="object",
        normalize_heatmap=True,
        resize_heatmap=True,
    ):
        """
        Computes the Grad-CAM heatmap for a target class. The heatmap represents spatial
        importance scores indicating which regions of the input most strongly contributed
        to the target prediction. Computation steps:

        1. Compute gradients of the target score with respect to feature maps.
        2. Perform global average pooling over gradients to obtain channel weights.
        3. Compute weighted feature map combination.
        4. Apply ReLU to retain only positive contributions.
        5. Optionally normalize the heatmap between ``0`` and ``1``.
        6. Optionally resize the heatmap to match the input spatial dimensions.

        Args:
            input_tensor: The input tensor (image or volume). Supported shapes ``(B, H, W, C)``
                for 2D and ``(B, D, H, W, C)`` for 3D.
            target_class_index (int or None, optional): Index of the target class for heatmap generation.
                If ``None``, the predicted class may be selected automatically depending on
                the parent implementation. For segmentation models with more than
                one output channel, this argument is required.
            mask_type: Type of mask to apply during segmentation target calculation

                - ``object``: Focuses the gradient calculation only on the
                    **predicted pixels/voxels** belonging to the `target_class_index`.
                    This is the standard approach to highlight the detected object
                    when classes are mutually exclusive.
                - ``all``: Calculates the gradient based on the **sum of all
                    predicted scores** for the `target_class_index` across the entire
                    spatial domain. This provides a global importance map for the
                    class and is often a practical choice for overlapping
                    multi-label segmentation channels.
                - ``single``: Calculates the gradient based only on the score of the
                    **single pixel/voxel** that has the maximum prediction value
                    for the `target_class_index`.
            normalize_heatmap: If ``True``, scales the heatmap values between ``0`` and ``1`` after ``ReLU``.
                Default: ``True``.
            resize_heatmap: If True, upsamples the heatmap to the input_tensor's spatial
                dimensions ``([D], H, W)``. Default: ``True``.

        Examples:
            .. code-block:: python

                import numpy as np
                from medicai.models import EfficientNetB0
                from medicai.utils import GradCAM

                model = EfficientNetB0(
                    input_shape=(224, 224, 3),
                )
                x = np.random.randn(1, 224, 224, 3)

                cam = GradCAM(
                    model=model,
                    target_layer="top_conv"
                )
                heatmap = cam.compute_heatmap(
                    input_tensor=x,
                    target_class_index=207
                )
                print(heatmap.shape) # (1, 224, 224, 1)

        .. code-block:: python

            import numpy as np
            from medicai.models import EfficientNetB0
            from medicai.utils import GradCAM

            model = EfficientNetB0(
                input_shape=(16, 128, 128, 1),
            )
            x = np.random.randn(1, 16, 128, 128, 1)

            cam = GradCAM(
                model=model,
                target_layer="top_conv"
            )
            heatmap = cam.compute_heatmap(
                input_tensor=x,
                target_class_index=207
            )
            print(heatmap.shape)


        .. code-block:: python

            import numpy as np
            from medicai.models import UNet
            from medicai.utils import GradCAM

            model = UNet(
                encoder_name='densenet121',
                input_shape=(64, 128, 128, 1),
                num_classes=5,
                classifier_activation="softmax",
            )
            x = np.random.randn(1, 64, 128, 128, 1)

            cam = GradCAM(
                model=model,
                target_layer="decoder_stage1_conv_2_activation"
            )
            heatmap = cam.compute_heatmap(
                input_tensor=x,
                target_class_index=3
            )
            print(heatmap.shape) # (1, 64, 128, 128, 1)

        .. code-block:: python

            import numpy as np
            from medicai.models import TransUNet
            from medicai.utils import GradCAM, extract_patches, merge_patches

            model = TransUNet(
                encoder_name="seresnext50",
                input_shape=(96, 96, 96, 4),
                num_classes=3,
                classifier_activation=None,
            )

            cam = GradCAM(
                model=model,
                target_layer="decoder_act_1",
            )

            x = np.random.randn(1, 128, 160, 160, 4).astype("float32")
            padded_inputs, info = extract_patches(
                inputs=x,
                roi_size=(96, 96, 96),
                overlap=0.5,
                mode="gaussian",
            )

            def cam_patch_generator():
                for slice_idx in info["slices"]:
                    patch = padded_inputs[(slice(None),) + slice_idx + (slice(None),)]
                    heatmap = cam.compute_heatmap(
                        input_tensor=patch,
                        target_class_index=2,
                        mask_type="all",
                        normalize_heatmap=False,
                        resize_heatmap=True,
                    )
                    yield heatmap, [slice_idx], info["importance_map"]

            full_heatmap = merge_patches(
                patch_generator=cam_patch_generator(),
                info=info,
                num_classes=1,
            )
            print(full_heatmap.shape)  # (1, 128, 160, 160, 1)

        Returns:
            The final heatmap as a NumPy array shaped
            ``(Batch, H, W, 1)`` or ``(Batch, [D,] H, W, 1)``. When
            ``normalize_heatmap=True``, each sample is scaled independently to
            the range ``[0, 1]`` after ``ReLU``.

        References:
            - Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
              `arXiv:1610.02391 <https://arxiv.org/abs/1610.02391>`_
            - Towards Interpretable Semantic Segmentation via Gradient-weighted Class Activation Mapping.
              `arXiv:2002.11434 <https://arxiv.org/abs/2002.11434>`_
        """
        # Ensure mask_type is valid before processing
        mask_type = mask_type.value if isinstance(mask_type, MaskType) else str(mask_type)
        if mask_type not in MaskType.values():
            raise ValueError(
                f"Unsupported mask_type '{mask_type}'. Use one of: {MaskType.values()}"
            )

        # 1. Compute gradients of the target score w.r.t. feature maps
        grads, hidden_output, _ = self.compute_gradients(
            input_tensor, target_class_index, mask_type
        )

        if grads is None:
            raise ValueError("Gradient computation failed - no gradients returned")

        grads = ops.cast(grads, "float32")
        hidden_output = ops.cast(hidden_output, "float32")

        # 2. Global Average Pooling of gradients (calculating the importance weight alpha_c^k)
        # The pooling axes are all spatial dimensions (1 to rank - 2)
        spatial_axes = tuple(range(1, len(grads.shape) - 1))
        pooled_grads = ops.mean(grads, axis=spatial_axes, keepdims=True)

        # 3. Weighted combination of feature maps
        # Multiply each feature map by its corresponding weight (pooled_grads) and sum across
        # the channel axis
        heatmap = ops.sum(hidden_output * pooled_grads, axis=-1)

        # 4. Apply ReLU: Only positive influence on the target class is considered
        heatmap = ops.relu(heatmap)

        if normalize_heatmap:
            # 5. Normalize heatmap between 0 and 1
            # Find the maximum value across all spatial dimensions
            heatmap_max = ops.max(heatmap, axis=tuple(range(1, len(heatmap.shape))), keepdims=True)
            eps = keras.config.epsilon()  # will add epsilon to avoid division by zero
            heatmap = ops.where(
                heatmap_max > 0, heatmap / (heatmap_max + eps), ops.zeros_like(heatmap)
            )

        # Add the channel dimension
        heatmap = ops.expand_dims(heatmap, axis=-1)

        if resize_heatmap:
            # 6. Resize to original spatial dimensions
            heatmap = self.resize_heatmap(heatmap, tuple(input_tensor.shape[1:-1]))

        return ops.convert_to_numpy(heatmap)
