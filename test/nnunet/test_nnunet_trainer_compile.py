from types import SimpleNamespace

import pytest

from medicai.trainer.nnunet.training.trainer import nnUNetTrainer


def _make_trainer(*, deep_supervision=True):
    trainer = object.__new__(nnUNetTrainer)
    trainer.cfg = SimpleNamespace(deep_supervision=deep_supervision)
    trainer.net_cfg = SimpleNamespace(deep_supervision=deep_supervision, n_pooling=3)
    return trainer


def test_flat_metrics_target_final_deep_supervision_output():
    trainer = _make_trainer()
    metric = object()

    result = trainer._metrics_for_model_outputs([metric])

    assert result == {"final": [metric], "aux_0": [], "aux_1": []}


def test_metric_mapping_preserves_requested_outputs_and_fills_others():
    trainer = _make_trainer()
    final_metric = object()
    aux_metric = object()

    result = trainer._metrics_for_model_outputs(
        {"final": [final_metric], "aux_0": [aux_metric]}
    )

    assert result == {
        "final": [final_metric],
        "aux_0": [aux_metric],
        "aux_1": [],
    }


def test_metric_mapping_rejects_unknown_model_outputs():
    trainer = _make_trainer()

    with pytest.raises(ValueError, match="unknown model output"):
        trainer._metrics_for_model_outputs({"unknown": [object()]})


def test_metrics_remain_unchanged_without_deep_supervision():
    trainer = _make_trainer(deep_supervision=False)
    metrics = [object()]

    assert trainer._metrics_for_model_outputs(metrics) is metrics
