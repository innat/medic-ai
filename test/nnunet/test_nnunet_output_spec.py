import keras
import pytest

from medicai.trainers.nnunet import NetworkContext, OutputSpec
from medicai.trainers.nnunet.models.dynamic_unet import build_unet_from_config
from medicai.trainers.nnunet.training.trainer import nnUNetTrainer
from medicai.trainers.nnunet.utils.config import NetworkConfig, TrainingConfig, nnUNetPlan


def _plan(*, n_classes=2, deep_supervision=True):
    config = NetworkConfig(
        spatial_dims=2,
        patch_size=[16, 16],
        n_pooling=2,
        base_filters=4,
        max_filters=8,
        pool_op_kernel_sizes=[[2, 2], [2, 2]],
        deep_supervision=deep_supervision,
        n_classes=n_classes,
        n_modalities=1,
    )
    return nnUNetPlan(
        network_type="2d",
        plan_2d=config,
        selected_configuration="2d",
    ), config


def _trainer(plan, config, tmp_path, trainer_class=nnUNetTrainer):
    trainer = trainer_class(
        model=build_unet_from_config(config),
        train_dataset=None,
        val_dataset=None,
        plan=plan,
        train_config=TrainingConfig(checkpoint_dir=str(tmp_path)),
        configuration="2d",
        auto_compile=False,
    )
    context = NetworkContext(plan=plan, plan_config=config, configuration="2d")
    trainer.model = trainer.create_network(context)
    return trainer


def test_default_network_output_spec_describes_deep_supervision(tmp_path):
    plan, config = _plan()
    trainer = _trainer(plan, config, tmp_path)

    output_spec = trainer.derive_output_spec()

    assert isinstance(output_spec, OutputSpec)
    assert output_spec.names == ("final", "aux_0")
    assert output_spec.channels == {"final": 2, "aux_0": 2}
    assert output_spec.scales["final"] == (1.0, 1.0)
    assert output_spec.scales["aux_0"] == (0.5, 0.5)
    assert output_spec.target_encoding == "categorical"


def test_custom_network_hook_can_replace_the_default_network(tmp_path):
    plan, config = _plan(deep_supervision=False)

    class PointwiseTrainer(nnUNetTrainer):
        def create_network(self, network_context):
            inputs = keras.Input(shape=(*network_context.plan_config.patch_size, 1))
            outputs = keras.layers.Conv2D(
                network_context.plan_config.n_classes,
                kernel_size=1,
                activation="softmax",
            )(inputs)
            return keras.Model(inputs, outputs)

    trainer = _trainer(plan, config, tmp_path, PointwiseTrainer)
    output_spec = trainer.derive_output_spec()

    assert output_spec.names == ("final",)
    assert output_spec.channels["final"] == 2
    assert trainer.model.layers[-1].__class__.__name__ == "Conv2D"


def test_trainer_recipe_preserves_constructor_options_for_runtime(tmp_path):
    plan, config = _plan(deep_supervision=False)

    class ConfiguredTrainer(nnUNetTrainer):
        def __init__(self, bottleneck_width):
            super().__init__()
            self.bottleneck_width = bottleneck_width

    recipe = ConfiguredTrainer(bottleneck_width=384)
    runtime = recipe.create_runtime(
        model=build_unet_from_config(config),
        train_dataset=None,
        val_dataset=None,
        plan=plan,
        train_config=TrainingConfig(checkpoint_dir=str(tmp_path)),
        configuration="2d",
    )

    assert runtime is not recipe
    assert runtime.bottleneck_width == 384
    assert runtime._recipe_mode is False


def test_output_spec_rejects_network_with_wrong_final_channels(tmp_path):
    plan, config = _plan(n_classes=2, deep_supervision=False)

    class WrongOutputTrainer(nnUNetTrainer):
        def create_network(self, network_context):
            inputs = keras.Input(shape=(*network_context.plan_config.patch_size, 1))
            outputs = keras.layers.Conv2D(3, kernel_size=1, activation="softmax")(inputs)
            return keras.Model(inputs, outputs)

    trainer = _trainer(plan, config, tmp_path, WrongOutputTrainer)

    with pytest.raises(ValueError, match="final output has 3 channels"):
        trainer.derive_output_spec()
