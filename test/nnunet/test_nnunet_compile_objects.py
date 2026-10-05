import keras

from medicai.trainers.nnunet import PerFoldFactory
from medicai.trainers.nnunet.training.specs import clone_callbacks, clone_compile_value


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
    assert cloned[0].get_config() == callback.get_config()
