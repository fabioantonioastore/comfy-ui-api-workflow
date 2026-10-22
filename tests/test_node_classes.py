import inspect
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

import pytest

from comfy_ui_api_workflow._impl.errors.node import InvalidNodeParameterValue
from comfy_ui_api_workflow._impl.nodes.validation import accepts_link
from comfy_ui_api_workflow._impl.workflow import _node_types

# These tests read only the classes, so they run without new_object_info.json.
# The checks against ComfyUI itself live in test_object_info_nodes.py.
LINK = ("Upstream_0", 0)
NODE_TYPES = sorted(_node_types().values(), key=lambda node_type: node_type.CLASS_TYPE)
# Combos whose options depend on another input; test_dynamic_nodes.py covers them.
DEPENDENT_COMBOS = {
    ("SaveVideo", "format_codec_encoding"),
    ("SaveVideo", "codec_encoding"),
}
# Choices that open the branches where every SaveVideo input exists.
WHOLE_TREE = {
    "SaveVideo": {
        "format_codec": "h264",
        "format_codec_encoding": "re-encode",
        "codec": "h264",
        "codec_encoding": "re-encode",
    }
}


def branch_inputs(node_type):
    # Inputs of dynamic combo branches and group rows can't all be set at
    # once; test_dynamic_object_info_nodes.py goes through them branch by branch.
    found = set()
    for name, spec in getattr(node_type, "DYNAMIC_INPUTS", {}).items():
        found.add(name)
        if hasattr(spec, "attributes"):
            found |= spec.attributes()
        else:
            found |= {
                spec.attribute(name, row, field)
                for row in range(spec.rows)
                for field in spec.fields
            }
    return found


def parameters(node_type):
    return [
        parameter
        for name, parameter in inspect.signature(node_type).parameters.items()
        if name != "node_id"
    ]


def choices(annotation):
    options = (
        get_args(annotation)
        if get_origin(annotation) in (Union, UnionType)
        else (annotation,)
    )
    return next(
        (get_args(option) for option in options if get_origin(option) is Literal), ()
    )


# Each input gets its own link, since positive and negative cannot share one.
def link_for(name):
    return (f"Upstream_{name}", 0)


def build(node_type, **values):
    required = {
        parameter.name: link_for(parameter.name)
        for parameter in parameters(node_type)
        if parameter.default is parameter.empty
    }
    return node_type(node_id=f"{node_type.__name__}_0", **{**required, **values})


def class_type(node_type):
    return node_type.CLASS_TYPE


@pytest.mark.parametrize("node_type", NODE_TYPES, ids=class_type)
def test_node_class_builds_with_its_defaults(node_type):
    node = build(node_type)
    inputs = node.inputs()
    fixed = {
        parameter.name: (
            link_for(parameter.name)
            if parameter.default is parameter.empty
            else parameter.default
        )
        for parameter in parameters(node_type)
        if parameter.default is not None
    }

    # A None default either leaves the input out or stands for a mutable
    # default, like a dict, that the constructor fills in; it is never sent as None.
    assert {name: inputs.get(name) for name in fixed} == fixed
    assert set(inputs) <= {parameter.name for parameter in parameters(node_type)}
    assert None not in inputs.values()


@pytest.mark.parametrize("node_type", NODE_TYPES, ids=class_type)
def test_every_input_that_takes_a_link_is_sent_as_one(node_type):
    linked = {
        parameter.name: link_for(parameter.name)
        for parameter in parameters(node_type)
        if (accepts_link(parameter.annotation) or parameter.annotation is Any)
        and parameter.name not in branch_inputs(node_type)
    }

    node = build(node_type, **WHOLE_TREE.get(node_type.CLASS_TYPE, {}), **linked)
    sent = node.get_comfy_ui_workflow_format()[node.node_id]

    assert sent["class_type"] == node_type.CLASS_TYPE
    assert {
        name: sent["inputs"][node_type.INPUT_NAMES.get(name, name)] for name in linked
    } == linked


COMBO_INPUTS = [
    pytest.param(
        node_type, parameter.name, id=f"{node_type.CLASS_TYPE}-{parameter.name}"
    )
    for node_type in NODE_TYPES
    for parameter in parameters(node_type)
    if choices(parameter.annotation)
    and (node_type.CLASS_TYPE, parameter.name) not in DEPENDENT_COMBOS
    and parameter.name not in branch_inputs(node_type)
]


@pytest.mark.parametrize(("node_type", "name"), COMBO_INPUTS)
def test_combo_input_takes_each_option_and_rejects_any_other(node_type, name):
    options = choices(inspect.signature(node_type).parameters[name].annotation)

    for option in options:
        assert build(node_type, **{name: option}).inputs()[name] == option
    with pytest.raises(
        InvalidNodeParameterValue, match=f"{name}|{name.replace('_', ' ')}"
    ):
        build(node_type, **{name: "__not_an_option__"})


def test_library_input_with_a_mutable_default_cannot_be_deleted():
    node = build(_node_types()["CurveEditor"])
    before = node.inputs()

    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        del node.curve

    assert str(excinfo.value) == "curve cannot be removed because it is not optional"
    assert node.inputs() == before


CONDITIONED_NODE_TYPES = [
    node_type
    for node_type in NODE_TYPES
    if {"positive", "negative"} <= set(inspect.signature(node_type).parameters)
]


@pytest.mark.parametrize("node_type", CONDITIONED_NODE_TYPES, ids=class_type)
def test_positive_and_negative_cannot_use_the_same_link(node_type):
    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        build(node_type, positive=LINK, negative=LINK)

    assert str(excinfo.value) == "positive and negative must not use the same input"
