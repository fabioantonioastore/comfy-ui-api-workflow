import pytest

from comfy_ui_api_workflow._impl.errors.node import InvalidNodeParameterValue
from comfy_ui_api_workflow._impl.exceptions import NodeException
from comfy_ui_api_workflow._impl.nodes.model.sampling import (
    KSAMPLER_SAMPLER_NAMES,
    KSAMPLER_SCHEDULERS,
    KSampler,
)


def make_ksampler(inputs, **overrides):
    return KSampler(node_id="KSampler_0", **{**inputs, **overrides})


def test_comfy_ui_workflow_format_with_defaults(ksampler_inputs):
    node = make_ksampler(ksampler_inputs)

    assert node.get_comfy_ui_workflow_format() == {
        "KSampler_0": {
            "class_type": "KSampler",
            "inputs": {
                "model": ("CheckpointLoaderSimple_0", 0),
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": ("CLIPTextEncode_1", 0),
                "negative": ("CLIPTextEncode_2", 0),
                "latent_image": ("EmptyLatentImage_3", 0),
                "seed": 0,
                "steps": 20,
                "cfg": 8.0,
                "denoise": 1.0,
            },
        }
    }


def test_output_exposes_latent(ksampler_inputs):
    node = make_ksampler(ksampler_inputs)

    assert node.output["LATENT"] == ("KSampler_0", 0)


def test_category_matches_comfy_ui():
    assert KSampler.CATEGORY == "model/sampling"


def test_scheduler_attribute_is_named_scheduler(ksampler_inputs):
    node = make_ksampler(ksampler_inputs, scheduler="karras")

    assert node.scheduler == "karras"


def inputs_of(node):
    return node.get_comfy_ui_workflow_format()[node.node_id]["inputs"]


@pytest.mark.parametrize("sampler_name", sorted(KSAMPLER_SAMPLER_NAMES))
def test_accepts_every_sampler_name(ksampler_inputs, sampler_name):
    node = make_ksampler(ksampler_inputs, sampler_name=sampler_name)

    assert inputs_of(node)["sampler_name"] == sampler_name


@pytest.mark.parametrize("scheduler", sorted(KSAMPLER_SCHEDULERS))
def test_accepts_every_scheduler(ksampler_inputs, scheduler):
    node = make_ksampler(ksampler_inputs, scheduler=scheduler)

    assert inputs_of(node)["scheduler"] == scheduler


@pytest.mark.parametrize(
    ("name", "choices"),
    [("sampler_name", KSAMPLER_SAMPLER_NAMES), ("scheduler", KSAMPLER_SCHEDULERS)],
)
def test_rejects_unknown_choice_listing_the_sorted_options(ksampler_inputs, name, choices):
    with pytest.raises(InvalidNodeParameterValue) as error:
        make_ksampler(ksampler_inputs, **{name: "not_an_option"})

    assert str(error.value) == (
        f"Invalid {name}, must be one of the following values: {sorted(choices)}"
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("seed", 0),
        ("seed", 18_446_744_073_709_551_615),
        ("steps", 1),
        ("steps", 10_000),
        ("cfg", 0.0),
        ("cfg", 100.0),
        ("denoise", 0.0),
        ("denoise", 1.0),
    ],
)
def test_accepts_range_boundaries(ksampler_inputs, name, value):
    node = make_ksampler(ksampler_inputs, **{name: value})

    assert inputs_of(node)[name] == value


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("seed", -1),
        ("seed", 18_446_744_073_709_551_616),
        ("steps", 0),
        ("steps", 10_001),
        ("cfg", -0.1),
        ("cfg", 100.1),
        ("denoise", -0.01),
        ("denoise", 1.01),
    ],
)
def test_rejects_out_of_range_values(ksampler_inputs, name, value):
    with pytest.raises(InvalidNodeParameterValue, match=name):
        make_ksampler(ksampler_inputs, **{name: value})


@pytest.mark.parametrize("name", ["cfg", "denoise"])
def test_rejects_nan_for_float_parameter(ksampler_inputs, name):
    with pytest.raises(InvalidNodeParameterValue, match=f"{name} must not be NaN"):
        make_ksampler(ksampler_inputs, **{name: float("nan")})


@pytest.mark.parametrize("name", ["seed", "steps", "cfg", "denoise"])
def test_rejects_bool_for_numeric_parameter(ksampler_inputs, name):
    with pytest.raises(InvalidNodeParameterValue, match=name):
        make_ksampler(ksampler_inputs, **{name: True})


@pytest.mark.parametrize(
    "name", ["seed", "steps", "cfg", "denoise", "sampler_name", "scheduler"]
)
def test_widgets_accept_links_from_other_nodes(ksampler_inputs, name):
    node = make_ksampler(ksampler_inputs, **{name: ("PrimitiveInt_9", 0)})

    assert inputs_of(node)[name] == ("PrimitiveInt_9", 0)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("steps", "20", "steps must be an int or a link, not str"),
        ("seed", 1.5, "seed must be an int or a link, not float"),
        ("cfg", None, "cfg must be a number or a link, not NoneType"),
        ("sampler_name", ["euler"], "sampler_name must be a str or a link, not list"),
        ("model", "v1-5-pruned-emaonly.safetensors", "model must be a link, not str"),
    ],
)
def test_rejects_values_of_another_type(ksampler_inputs, name, value, message):
    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        make_ksampler(ksampler_inputs, **{name: value})

    assert str(excinfo.value) == message


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("steps", 0, "steps must be in closed range: [1, 10000]"),
        ("cfg", 100.5, "cfg must be in closed range: [0.0, 100.0]"),
        ("scheduler", "not_a_scheduler", "Invalid scheduler"),
    ],
)
def test_change_to_an_invalid_value_keeps_the_previous_value(
    ksampler_inputs, name, value, message
):
    node = make_ksampler(ksampler_inputs)
    before = inputs_of(node)

    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        setattr(node, name, value)

    assert str(excinfo.value).startswith(message)
    assert inputs_of(node) == before


def test_invalid_parameter_is_node_exception():
    assert issubclass(InvalidNodeParameterValue, NodeException)
