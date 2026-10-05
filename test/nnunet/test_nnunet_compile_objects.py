import json
from types import SimpleNamespace

import keras
import pytest

from medicai.trainers.nnunet import OutputSpec, PerFoldFactory, nnUNetPipeline
from medicai.trainers.nnunet.training.specs import clone_callbacks, clone_compile_value
from medicai.trainers.nnunet.training.trainer import nnUNetTrainer


def test_registered_optimizer_is_cloned_for_a_new_fold():
    optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    cloned = clone_compile_value(optimizer)

    assert cloned is not optimizer
    assert cloned.get_config() == optimizer.get_config()


def test_per_fold_factory_creates_independent_custom_objects():
    created = []

    def factory():
        value = object()
        created.append(value)
        return value

    per_fold = PerFoldFactory(factory)
    first = clone_compile_value(per_fold)
    second = clone_compile_value(per_fold)

    assert first is not second
    assert created == [first, second]


def test_callbacks_are_not_reused_between_training_runs():
    callback = keras.callbacks.EarlyStopping(patience=2)

    cloned = clone_callbacks([callback])

    assert cloned[0] is not callback
    assert cloned[0].patience == callback.patience


def test_checkpoint_provenance_accepts_matching_recipe(tmp_path):
    checkpoint = tmp_path / "fold_0" / "best_model.weights.h5"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    output_spec = OutputSpec(
        names=("final",),
        scales={"final": (1.0, 1.0)},
        channels={"final": 2},
        activation="softmax",
        target_encoding="categorical",
    )
    trainer = SimpleNamespace(
        configuration="2d",
        architecture={"family": "test"},
        output_spec=output_spec,
        _json_safe=nnUNetTrainer._json_safe,
    )
    plan = SimpleNamespace(dataset_name="demo", network_type="2d")
    identity = f"{type(trainer).__module__}.{type(trainer).__qualname__}"
    provenance = {
        "trainer": identity,
        "dataset": "demo",
        "network": "2d",
        "configuration": "2d",
        "fold": 0,
        "architecture": {"family": "test"},
        "output_spec": output_spec.to_dict(),
    }
    with (checkpoint.parent / "training_run.json").open("w", encoding="utf-8") as stream:
        json.dump(provenance, stream)

    nnUNetPipeline._validate_checkpoint_provenance(
        checkpoint_path=checkpoint,
        trainer=trainer,
        plan=plan,
        fold=0,
    )


def test_checkpoint_provenance_rejects_trainer_mismatch(tmp_path):
    checkpoint = tmp_path / "fold_0" / "best_model.weights.h5"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    (checkpoint.parent / "training_run.json").write_text(
        json.dumps({"trainer": "different.Trainer"}),
        encoding="utf-8",
    )
    trainer = SimpleNamespace(
        configuration="2d",
        architecture={},
        output_spec=None,
        _json_safe=nnUNetTrainer._json_safe,
    )
    plan = SimpleNamespace(dataset_name="demo", network_type="2d")

    with pytest.raises(ValueError, match="trainer identity"):
        nnUNetPipeline._validate_checkpoint_provenance(
            checkpoint_path=checkpoint,
            trainer=trainer,
            plan=plan,
            fold=0,
        )
