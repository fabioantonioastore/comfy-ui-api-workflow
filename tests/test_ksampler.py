import pytest

from comfy_ui_api_workflow._impl.enums import OutputTypes
from comfy_ui_api_workflow._impl.errors.node import InvalidNodeParameterValue
from comfy_ui_api_workflow._impl.errors.output import OutputNotFoundError
from comfy_ui_api_workflow._impl.exceptions import NodeException
from comfy_ui_api_workflow._impl.nodes.base import NodeOutput
from comfy_ui_api_workflow._impl.nodes.model.sampling import (
    KSAMPLER_SAMPLER_NAMES,
    KSAMPLER_SCHEDULERS,
    KSampler,
)


def make_ksampler(inputs, **overrides):
    return KSampler(node_id="KSampler0", **{**inputs, **overrides})


def test_comfy_ui_workflow_format_with_defaults(ksampler_inputs):
    node = make_ksampler(ksampler_inputs)

    assert node.get_comfy_ui_workflow_format() == {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,
            "steps": 20,
            "cfg": 8.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["Checkpoint0", 0],
            "positive": ["Positive1", 0],
            "negative": ["Negative2", 0],
            "latent_image": ["Latent3", 0],
        },
    }


def test_output_exposes_latent(ksampler_inputs):
    node = make_ksampler(ksampler_inputs)

    assert node.output.get(OutputTypes.LATENT) == ["KSampler0", 0]


def test_scheduler_attribute_is_named_scheduler(ksampler_inputs):
    node = make_ksampler(ksampler_inputs, scheduler="karras")

    assert node.scheduler == "karras"


@pytest.mark.parametrize("sampler_name", sorted(KSAMPLER_SAMPLER_NAMES))
def test_accepts_every_sampler_name(ksampler_inputs, sampler_name):
    node = make_ksampler(ksampler_inputs, sampler_name=sampler_name)

    assert node.get_comfy_ui_workflow_format()["inputs"]["sampler_name"] == sampler_name


@pytest.mark.parametrize("scheduler", sorted(KSAMPLER_SCHEDULERS))
def test_accepts_every_scheduler(ksampler_inputs, scheduler):
    node = make_ksampler(ksampler_inputs, scheduler=scheduler)

    assert node.get_comfy_ui_workflow_format()["inputs"]["scheduler"] == scheduler


def test_rejects_unknown_sampler_name(ksampler_inputs):
    with pytest.raises(InvalidNodeParameterValue, match="sampler name"):
        make_ksampler(ksampler_inputs, sampler_name="not_a_sampler")


def test_rejects_unknown_scheduler(ksampler_inputs):
    with pytest.raises(InvalidNodeParameterValue, match="scheduler"):
        make_ksampler(ksampler_inputs, scheduler="not_a_scheduler")


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

    assert node.get_comfy_ui_workflow_format()["inputs"][name] == value


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


@pytest.mark.parametrize("name", ["seed", "steps", "cfg", "denoise"])
def test_rejects_bool_for_numeric_parameter(ksampler_inputs, name):
    with pytest.raises(InvalidNodeParameterValue, match=name):
        make_ksampler(ksampler_inputs, **{name: True})


@pytest.mark.parametrize("name", ["model", "positive", "negative", "latent_image"])
def test_rejects_input_of_wrong_output_type(ksampler_inputs, name):
    wrong = NodeOutput(node_id="Wrong9", outputs={})

    with pytest.raises(OutputNotFoundError):
        make_ksampler(ksampler_inputs, **{name: wrong})


def test_invalid_parameter_is_node_exception():
    assert issubclass(InvalidNodeParameterValue, NodeException)
