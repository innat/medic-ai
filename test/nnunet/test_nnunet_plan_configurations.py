import pytest

from medicai.trainer.nnunet import nnUNetPipeline
from medicai.trainer.nnunet.utils.config import NetworkConfig, nnUNetPlan


def _plan(selected_configuration="2d"):
    return nnUNetPlan(
        network_type="2d",
        plan_2d=NetworkConfig(spatial_dims=2, patch_size=[128, 128]),
        selected_configuration=selected_configuration,
    )


def test_plan_exposes_available_configurations_by_name():
    plan = _plan()

    assert plan.configurations == {"2d": plan.plan_2d}
    assert plan.get_configuration("2d") is plan.plan_2d


def test_plan_rejects_unknown_configuration_with_available_choices():
    plan = _plan()

    with pytest.raises(ValueError, match="available configurations: 2d"):
        plan.get_configuration("3d_fullres")


def test_saved_selection_is_restored_by_a_new_pipeline_instance(tmp_path):
    plan = _plan()
    plan.to_json(tmp_path / "nnunet_plans.json")
    loaded_plan = nnUNetPlan.from_json(tmp_path / "nnunet_plans.json")
    pipeline = nnUNetPipeline(input_path=tmp_path)

    pipeline._sync_configuration(loaded_plan)

    assert pipeline.configuration == "2d"
