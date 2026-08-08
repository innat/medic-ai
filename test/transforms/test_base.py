import numpy as np
import pytest
import tensorflow as tf
import keras
from keras import ops

from medicai.transforms import (
    InvertibleTransform,
    KeyedTransform,
    LambdaTransform,
    RandomChoice,
    RandomTransform,
    TensorBundle,
    Transform,
)
from medicai.transforms.base import ensure_tensor_bundle
from medicai.transforms.utils import (
    LayoutInfo,
    ensure_batch_axis,
    ensure_spatial_tuple,
    get_spatial_rank,
    get_spatial_shape,
    normalize_axes,
    normalize_spatial_axes,
    restore_from_batch_axis,
    resolve_layout,
    resolve_spatial_axes,
    validate_input_mode,
    validate_layout,
    validate_spatial_rank,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_tensorbundle_creation_and_meta_access():
    data = {
        "image": as_tensor(np.zeros((4, 4, 1), dtype=np.float32)),
        "label": as_tensor(np.ones((4, 4, 1), dtype=np.int32)),
    }
    meta = {"pixdim": [1.0, 1.0, 1.0], "id": "case-a"}
    bundle = TensorBundle(data, meta)

    assert bundle.data["image"] is data["image"]
    assert bundle.meta["id"] == "case-a"
    assert bundle["pixdim"] == [1.0, 1.0, 1.0]


@pytest.mark.unit
def test_tensorbundle_validates_applied_transform_trace_type():
    bundle = TensorBundle({"image": as_tensor(np.zeros((2, 2, 1), dtype=np.float32))})
    bundle.meta["applied_transforms"] = "invalid"

    with pytest.raises(TypeError, match="must be a list"):
        bundle.get_applied_transforms()


@pytest.mark.unit
def test_tensorbundle_setitem_routes_to_data_or_meta():
    bundle = TensorBundle({"image": as_tensor(np.zeros((2, 2, 1), dtype=np.float32))})
    new_image = as_tensor(np.ones((2, 2, 1), dtype=np.float32))
    bundle["image"] = new_image
    bundle["affine"] = np.eye(4, dtype=np.float32).tolist()

    np.testing.assert_allclose(
        ops.convert_to_numpy(bundle["image"]),
        ops.convert_to_numpy(new_image),
    )
    assert "affine" in bundle.meta


@pytest.mark.unit
def test_tensorbundle_contains_checks_data_and_meta():
    bundle = TensorBundle(
        {"image": as_tensor(np.zeros((2, 2, 1), dtype=np.float32))},
        {"affine": np.eye(4, dtype=np.float32).tolist()},
    )

    assert "image" in bundle
    assert "affine" in bundle
    assert "missing" not in bundle


@pytest.mark.unit
def test_tensorbundle_missing_key_raises_keyerror():
    bundle = TensorBundle({"image": as_tensor(np.zeros((2, 2, 1), dtype=np.float32))})
    with pytest.raises(KeyError):
        _ = bundle["unknown"]


@pytest.mark.unit
def test_ensure_tensor_bundle_converts_numpy_inputs():
    bundle = ensure_tensor_bundle(
        {"image": np.ones((4, 4, 1), dtype=np.float32)},
        {"affine": np.eye(4, dtype=np.float32)},
    )

    assert isinstance(bundle, TensorBundle)
    assert hasattr(bundle["image"], "shape")
    assert hasattr(bundle["affine"], "shape")


@pytest.mark.unit
def test_ensure_tensor_bundle_converts_numpy_scalars():
    bundle = ensure_tensor_bundle(
        {"value": np.float32(3.5)},
        {"index": np.int32(2)},
    )

    assert hasattr(bundle["value"], "shape")
    assert hasattr(bundle["index"], "shape")


@pytest.mark.unit
def test_transform_base_normalizes_mapping_inputs():
    class AddOne(Transform):
        def apply(self, bundle):
            bundle["image"] = bundle["image"] + 1.0
            return bundle

    output = AddOne()({"image": np.zeros((2, 2, 1), dtype=np.float32)})
    np.testing.assert_allclose(ops.convert_to_numpy(output["image"]), 1.0)


@pytest.mark.unit
def test_keyed_transform_iter_present_keys_respects_missing_policy():
    class IdentityKeyedTransform(KeyedTransform):
        def apply(self, bundle):
            for key in self.iter_present_keys(bundle):
                bundle[key] = bundle[key]
            return bundle

    strict_transform = IdentityKeyedTransform(keys=["image", "label"])
    with pytest.raises(KeyError):
        strict_transform(TensorBundle({"image": as_tensor(np.zeros((2, 2, 1), dtype=np.float32))}))

    permissive_transform = IdentityKeyedTransform(keys=["image", "label"], allow_missing_keys=True)
    output = permissive_transform(
        TensorBundle({"image": as_tensor(np.zeros((2, 2, 1), dtype=np.float32))})
    )
    assert "image" in output.data


@pytest.mark.unit
def test_keyed_transform_apply_to_present_keys_updates_only_available_keys():
    class AddPerKey(KeyedTransform):
        def apply(self, bundle):
            self.apply_to_present_keys(bundle, lambda tensor, _: tensor + 2.0)
            return bundle

    bundle = TensorBundle({"image": as_tensor(np.zeros((2, 2, 1), dtype=np.float32))})
    output = AddPerKey(keys=["image", "label"], allow_missing_keys=True)(bundle)

    np.testing.assert_allclose(ops.convert_to_numpy(output["image"]), 2.0)


@pytest.mark.unit
def test_invertible_transform_records_trace_entries():
    class TraceOnlyTransform(InvertibleTransform):
        def apply(self, bundle):
            return self.record_transform(bundle, {"axes": (0, 1), "k": 1})

        def inverse(self, bundle):
            return bundle

    bundle = TraceOnlyTransform()(TensorBundle({"image": as_tensor(np.zeros((2, 2, 1)))}))
    trace = bundle.get_applied_transforms()

    assert len(trace) == 1
    assert trace[0]["name"] == "TraceOnlyTransform"
    assert trace[0]["params"]["k"] == 1
    assert trace[0]["applied"] is True
    assert trace[0]["random"] is False
    assert trace[0]["invertible"] is True


@pytest.mark.unit
def test_random_transform_validates_probability_range():
    class DummyRandomTransform(RandomTransform):
        def apply(self, bundle):
            return bundle

    DummyRandomTransform(prob=0.0)
    DummyRandomTransform(prob=1.0)

    with pytest.raises(ValueError):
        DummyRandomTransform(prob=-0.1)

    with pytest.raises(ValueError):
        DummyRandomTransform(prob=1.1)


@pytest.mark.unit
def test_random_transform_builds_standardized_trace_entries():
    class DummyRandomTransform(RandomTransform):
        def apply(self, bundle):
            return self.record_random_transform(
                bundle,
                params={"value": 3},
                applied=False,
                kernel="DummyKernel",
            )

    bundle = DummyRandomTransform(prob=0.5)(TensorBundle({"image": as_tensor(np.zeros((2, 2, 1)))}))
    trace = bundle.get_applied_transforms()[-1]

    assert trace["name"] == "DummyRandomTransform"
    assert trace["params"]["value"] == 3
    assert trace["applied"] is False
    assert trace["random"] is True
    assert trace["invertible"] is False
    assert trace["kernel"] == "DummyKernel"


@pytest.mark.unit
def test_random_transform_accepts_integer_seed_contract():
    class SeedAwareRandomTransform(RandomTransform):
        def apply(self, bundle):
            bundle["should_apply"] = self.sample_should_apply()
            return bundle

    transform = SeedAwareRandomTransform(prob=0.5, seed=13)
    out = transform(TensorBundle({"image": as_tensor(np.zeros((2, 2, 1), dtype=np.float32))}))

    assert "should_apply" in out.data


@pytest.mark.unit
def test_random_transform_accepts_seed_generator_contract():
    class SeedAwareRandomTransform(RandomTransform):
        def apply(self, bundle):
            bundle["should_apply"] = self.sample_should_apply()
            return bundle

    seed = keras.random.SeedGenerator(13)
    transform = SeedAwareRandomTransform(prob=0.5, seed=seed)
    out = transform(TensorBundle({"image": as_tensor(np.zeros((2, 2, 1), dtype=np.float32))}))

    assert "should_apply" in out.data


@pytest.mark.unit
def test_random_transform_replays_deterministically_with_same_integer_seed():
    class SeedAwareRandomTransform(RandomTransform):
        def apply(self, bundle):
            bundle["should_apply"] = self.sample_should_apply()
            return bundle

    image = as_tensor(np.zeros((2, 2, 1), dtype=np.float32))
    first = SeedAwareRandomTransform(prob=0.5, seed=7)(TensorBundle({"image": image}))
    second = SeedAwareRandomTransform(prob=0.5, seed=7)(TensorBundle({"image": image}))

    assert bool(ops.convert_to_numpy(first["should_apply"])) == bool(
        ops.convert_to_numpy(second["should_apply"])
    )


@pytest.mark.unit
def test_random_transform_rejects_unsupported_seed_type():
    class SeedAwareRandomTransform(RandomTransform):
        def apply(self, bundle):
            bundle["should_apply"] = self.sample_should_apply()
            return bundle

    with pytest.raises(TypeError, match="`seed` must be None, an integer, or keras.random.SeedGenerator"):
        SeedAwareRandomTransform(prob=0.5, seed="bad-seed")


@pytest.mark.unit
def test_random_choice_validates_configuration():
    with pytest.raises(ValueError, match="at least one transform"):
        RandomChoice(transforms=[], num_choices=1)

    with pytest.raises(ValueError, match="cannot request more than 1 transforms"):
        RandomChoice(transforms=[lambda bundle: bundle], num_choices=2)

    with pytest.raises(ValueError, match="same length as `transforms`"):
        RandomChoice(transforms=[lambda bundle: bundle], num_choices=1, weights=[1.0, 2.0])

    with pytest.raises(ValueError, match="non-negative"):
        RandomChoice(transforms=[lambda bundle: bundle], num_choices=1, weights=[-1.0])

    with pytest.raises(ValueError, match="positive value"):
        RandomChoice(transforms=[lambda bundle: bundle], num_choices=0, weights=[0.0])


@pytest.mark.unit
def test_random_choice_records_eager_trace_metadata():
    image = as_tensor(np.ones((4, 4, 1), dtype=np.float32))
    choice = RandomChoice(
        transforms=[
            ShiftIntensity(keys=["image"], offset=1.0),
            Flip(keys=["image"], spatial_axis=1),
        ],
        num_choices=1,
        prob=1.0,
        seed=5,
    )

    output = choice(TensorBundle({"image": image}))
    trace = output.get_applied_transforms()[-1]

    assert trace["name"] == "RandomChoice"
    assert trace["random"] is True
    assert trace["kernel"] == "RandomChoice"
    assert trace["params"]["num_selected"] == 1
    assert trace["params"]["num_choices"] == (1, 1)
    assert len(trace["params"]["selected_indices"]) == 1
    assert len(trace["params"]["selected_names"]) == 1


@pytest.mark.unit
def test_random_choice_inverse_replays_selected_invertible_transforms():
    image = as_tensor(np.arange(12, dtype=np.float32).reshape(3, 4, 1))
    choice = RandomChoice(
        transforms=[
            Flip(keys=["image"], spatial_axis=1),
            ShiftIntensity(keys=["image"], offset=2.0),
        ],
        num_choices=2,
        prob=1.0,
        seed=7,
    )

    forward = choice(TensorBundle({"image": image}))
    restored = choice.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_lambda_transform_applies_deterministic_callable_and_records_trace():
    bundle = TensorBundle({"image": as_tensor(np.ones((2, 2, 1), dtype=np.float32))})
    transform = LambdaTransform(
        keys=["image"],
        fn=lambda tensor: tensor + 2.0,
        name="add_two",
        trace_params={"value": 2},
    )

    output = transform(bundle)
    trace = output.get_applied_transforms()[-1]

    np.testing.assert_allclose(ops.convert_to_numpy(output["image"]), 3.0)
    assert trace["name"] == "LambdaTransform"
    assert trace["params"]["keys"] == ["image"]
    assert trace["params"]["value"] == 2
    assert trace["random"] is False
    assert trace["invertible"] is False
    assert trace["kernel"] == "add_two"


@pytest.mark.unit
def test_lambda_transform_supports_tensor_and_key_signature():
    bundle = TensorBundle({"image": as_tensor(np.ones((2, 2, 1), dtype=np.float32))})
    transform = LambdaTransform(
        keys=["image"],
        fn=lambda tensor, key: tensor + (1.0 if key == "image" else 0.0),
    )

    output = transform(bundle)

    np.testing.assert_allclose(ops.convert_to_numpy(output["image"]), 2.0)


@pytest.mark.unit
def test_lambda_transform_supports_random_application():
    image = as_tensor(np.ones((2, 2, 1), dtype=np.float32))

    skipped = LambdaTransform(keys=["image"], fn=lambda tensor: tensor + 3.0, prob=0.0)(
        TensorBundle({"image": image})
    )
    applied = LambdaTransform(keys=["image"], fn=lambda tensor: tensor + 3.0, prob=1.0)(
        TensorBundle({"image": image})
    )

    np.testing.assert_allclose(ops.convert_to_numpy(skipped["image"]), 1.0)
    np.testing.assert_allclose(ops.convert_to_numpy(applied["image"]), 4.0)
    assert not bool(ops.convert_to_numpy(skipped.get_applied_transforms()[-1]["applied"]))
    assert bool(ops.convert_to_numpy(applied.get_applied_transforms()[-1]["applied"]))


@pytest.mark.unit
def test_lambda_transform_supports_inverse_and_meta_hooks():
    image = as_tensor(np.ones((2, 2, 1), dtype=np.float32))
    transform = LambdaTransform(
        keys=["image"],
        fn=lambda tensor: tensor + 5.0,
        inverse_fn=lambda tensor: tensor - 5.0,
        meta_fn=lambda meta: {**meta, "forward_tag": True},
        inverse_meta_fn=lambda meta: {**meta, "inverse_tag": True},
    )

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(ops.convert_to_numpy(restored["image"]), 1.0)
    assert forward.meta["forward_tag"] is True
    assert restored.meta["inverse_tag"] is True


@pytest.mark.unit
def test_lambda_transform_skips_meta_hook_when_probabilistic_apply_is_false():
    image = as_tensor(np.ones((2, 2, 1), dtype=np.float32))
    transform = LambdaTransform(
        keys=["image"],
        fn=lambda tensor: tensor + 5.0,
        prob=0.0,
        meta_fn=lambda meta: {**meta, "forward_tag": True},
    )

    result = transform(TensorBundle({"image": image}))

    np.testing.assert_allclose(ops.convert_to_numpy(result["image"]), 1.0)
    assert "forward_tag" not in result.meta


@pytest.mark.unit
def test_lambda_transform_inverse_with_prob_uses_tensor_trace_flag():
    image = as_tensor(np.ones((2, 2, 1), dtype=np.float32))
    transform = LambdaTransform(
        keys=["image"],
        fn=lambda tensor: tensor + 5.0,
        inverse_fn=lambda tensor: tensor - 5.0,
        prob=1.0,
    )

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(ops.convert_to_numpy(restored["image"]), 1.0)


@pytest.mark.unit
def test_lambda_transform_probabilistic_meta_hooks_run_in_eager_mode_when_applied():
    image = as_tensor(np.ones((2, 2, 1), dtype=np.float32))
    transform = LambdaTransform(
        keys=["image"],
        fn=lambda tensor: tensor + 5.0,
        inverse_fn=lambda tensor: tensor - 5.0,
        meta_fn=lambda meta: {**meta, "forward_tag": True},
        inverse_meta_fn=lambda meta: {**meta, "inverse_tag": True},
        prob=1.0,
    )

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(ops.convert_to_numpy(restored["image"]), 1.0)
    assert forward.meta["forward_tag"] is True
    assert restored.meta["inverse_tag"] is True


@pytest.mark.unit
def test_lambda_transform_respects_missing_key_policy_and_prob_validation():
    with pytest.raises(ValueError):
        LambdaTransform(keys=["image"], fn=lambda tensor: tensor, prob=1.5)

    permissive = LambdaTransform(keys=["image"], fn=lambda tensor: tensor, allow_missing_keys=True)
    bundle = TensorBundle({"other": as_tensor(np.ones((2, 2, 1), dtype=np.float32))})
    assert permissive(bundle) is bundle

    strict = LambdaTransform(keys=["image"], fn=lambda tensor: tensor)
    with pytest.raises(KeyError):
        strict(bundle)


@pytest.mark.unit
def test_lambda_transform_accepts_builtin_tensor_functions_without_signature_errors():
    transform = LambdaTransform(keys=["image"], fn=tf.identity)
    image = as_tensor(np.ones((2, 2, 1), dtype=np.float32))

    result = transform(TensorBundle({"image": image}))

    np.testing.assert_allclose(ops.convert_to_numpy(result["image"]), 1.0)


@pytest.mark.unit
def test_lambda_transform_passes_key_to_varargs_callable():
    image = as_tensor(np.ones((2, 2, 1), dtype=np.float32))

    def fn(*args):
        tensor, key = args
        return tensor + (1.0 if key == "image" else 0.0)

    transform = LambdaTransform(keys=["image"], fn=fn)
    result = transform(TensorBundle({"image": image}))

    np.testing.assert_allclose(ops.convert_to_numpy(result["image"]), 2.0)


@pytest.mark.unit
def test_spatial_helpers_handle_2d_and_3d_channel_last_tensors():
    image_2d = as_tensor(np.zeros((16, 12, 1), dtype=np.float32))
    image_3d = as_tensor(np.zeros((8, 16, 12, 1), dtype=np.float32))

    assert get_spatial_rank(image_2d) == 2
    assert get_spatial_rank(image_3d) == 3
    assert validate_spatial_rank(image_2d) == 2
    assert validate_spatial_rank(image_3d) == 3
    assert ensure_spatial_tuple(4, 3, "roi_size") == (4, 4, 4)
    assert ensure_spatial_tuple((4, 5), 2, "roi_size") == (4, 5)
    assert normalize_axes((-1, 0), rank=3) == (2, 0)
    assert normalize_spatial_axes((-1, 0), spatial_rank=3) == (2, 0)
    np.testing.assert_array_equal(ops.convert_to_numpy(get_spatial_shape(image_2d)), [16, 12])
    np.testing.assert_array_equal(ops.convert_to_numpy(get_spatial_shape(image_3d)), [8, 16, 12])


@pytest.mark.unit
def test_resolve_layout_describes_sample_and_batch_channel_last_tensors():
    sample_2d = as_tensor(np.zeros((16, 12, 1), dtype=np.float32))
    sample_3d = as_tensor(np.zeros((8, 16, 12, 1), dtype=np.float32))
    batch_2d = as_tensor(np.zeros((4, 16, 12, 1), dtype=np.float32))
    batch_3d = as_tensor(np.zeros((2, 8, 16, 12, 1), dtype=np.float32))

    layout_sample_2d = resolve_layout(sample_2d, input_mode="sample")
    layout_sample_3d = resolve_layout(sample_3d, input_mode="sample")
    layout_batch_2d = resolve_layout(batch_2d, input_mode="batch")
    layout_batch_3d = resolve_layout(batch_3d, input_mode="batch")

    assert isinstance(layout_sample_2d, LayoutInfo)
    assert layout_sample_2d == LayoutInfo(
        tensor_rank=3,
        spatial_rank=2,
        batched=False,
        batch_axis=None,
        channel_axis=2,
        spatial_axes=(0, 1),
    )
    assert layout_sample_3d == LayoutInfo(
        tensor_rank=4,
        spatial_rank=3,
        batched=False,
        batch_axis=None,
        channel_axis=3,
        spatial_axes=(0, 1, 2),
    )
    assert layout_batch_2d == LayoutInfo(
        tensor_rank=4,
        spatial_rank=2,
        batched=True,
        batch_axis=0,
        channel_axis=3,
        spatial_axes=(1, 2),
    )
    assert layout_batch_3d == LayoutInfo(
        tensor_rank=5,
        spatial_rank=3,
        batched=True,
        batch_axis=0,
        channel_axis=4,
        spatial_axes=(1, 2, 3),
    )

    np.testing.assert_array_equal(
        ops.convert_to_numpy(get_spatial_shape(batch_2d, input_mode="batch")),
        [16, 12],
    )
    np.testing.assert_array_equal(
        ops.convert_to_numpy(get_spatial_shape(batch_3d, input_mode="batch")),
        [8, 16, 12],
    )


@pytest.mark.unit
def test_validate_input_mode_and_layout_helpers_cover_sample_and_batch_contracts():
    sample_3d = as_tensor(np.zeros((8, 16, 12, 1), dtype=np.float32))
    batch_2d = as_tensor(np.zeros((4, 16, 12, 1), dtype=np.float32))

    assert validate_input_mode("sample") == "sample"
    assert validate_input_mode("batch") == "batch"
    assert validate_input_mode(
        "sample",
        supported_modes=("sample",),
        transform_name="Spacing",
    ) == "sample"

    sample_layout = validate_layout(
        sample_3d,
        input_mode="sample",
        allowed_spatial_ranks=(3,),
    )
    batch_layout = validate_layout(
        batch_2d,
        input_mode="batch",
        allowed_spatial_ranks=(2,),
    )

    assert sample_layout.spatial_axes == (0, 1, 2)
    assert batch_layout.spatial_axes == (1, 2)
    assert resolve_spatial_axes(sample_3d, (0, -1), input_mode="sample") == (0, 2)
    assert resolve_spatial_axes(batch_2d, (0, -1), input_mode="batch") == (1, 2)


@pytest.mark.unit
def test_batch_axis_helpers_normalize_sample_and_batch_inputs():
    sample_2d = as_tensor(np.zeros((4, 5, 1), dtype=np.float32))
    batch_2d = as_tensor(np.zeros((2, 4, 5, 1), dtype=np.float32))

    sample_batched, sample_added = ensure_batch_axis(sample_2d, input_mode="sample")
    batch_batched, batch_added = ensure_batch_axis(batch_2d, input_mode="batch")

    assert sample_added is True
    assert batch_added is False
    assert tuple(ops.shape(sample_batched)) == (1, 4, 5, 1)
    assert tuple(ops.shape(batch_batched)) == (2, 4, 5, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restore_from_batch_axis(sample_batched, sample_added)),
        ops.convert_to_numpy(sample_2d),
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(restore_from_batch_axis(batch_batched, batch_added)),
        ops.convert_to_numpy(batch_2d),
    )


@pytest.mark.unit
def test_spatial_helpers_validate_invalid_inputs():
    image_1d = as_tensor(np.zeros((8,), dtype=np.float32))

    with pytest.raises(ValueError):
        get_spatial_rank(image_1d)

    with pytest.raises(ValueError):
        validate_spatial_rank(as_tensor(np.zeros((2, 3, 4, 5, 1), dtype=np.float32)))

    with pytest.raises(ValueError):
        ensure_spatial_tuple((1, 2), 3, "roi_size")

    with pytest.raises(ValueError):
        normalize_axes((0, 0), rank=3)

    with pytest.raises(ValueError, match="input_mode"):
        resolve_layout(as_tensor(np.zeros((4, 4, 1), dtype=np.float32)), input_mode="unknown")

    with pytest.raises(ValueError, match="channel-last batch tensor"):
        resolve_layout(as_tensor(np.zeros((4, 4, 1), dtype=np.float32)), input_mode="batch")

    with pytest.raises(ValueError, match="supports only input_mode values"):
        validate_input_mode("batch", supported_modes=("sample",), transform_name="Spacing")

    with pytest.raises(ValueError, match="Expected spatial rank in"):
        validate_layout(
            as_tensor(np.zeros((4, 16, 12, 1), dtype=np.float32)),
            input_mode="batch",
            allowed_spatial_ranks=(3,),
        )
