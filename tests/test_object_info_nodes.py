import importlib
import json
import pkgutil
import re
from pathlib import Path
from typing import Any, NamedTuple

import pytest

import comfy_ui_api_workflow._impl.nodes as nodes_package
from comfy_ui_api_workflow._impl.errors.node import InvalidNodeParameterValue
from comfy_ui_api_workflow._impl.nodes.base import BaseNode

OBJECT_INFO = Path(__file__).resolve().parent.parent / "new_object_info.json"

if not OBJECT_INFO.exists():
    pytest.skip(
        f"{OBJECT_INFO.name} not found at the project root", allow_module_level=True
    )

SPECS = json.loads(OBJECT_INFO.read_text(encoding="utf-8"))
DYNAMIC_TYPES = {"COMFY_AUTOGROW_V3", "COMFY_DYNAMICCOMBO_V3", "COMFY_DYNAMICGROUP_V3"}
WIDGET_KINDS = {
    "INT": "int",
    "FLOAT": "float",
    "STRING": "string",
    "BOOLEAN": "boolean",
}
OUTPUT_TYPES = {
    part
    for spec in SPECS.values()
    for output in spec["output"]
    for part in str(output).split(",")
}
# Combos whose options come from the server running ComfyUI: files, folders, devices.
SERVER_COMBO_FLAGS = (
    "image_upload",
    "audio_upload",
    "video_upload",
    "file_upload",
    "image_folder",
    "remote",
)
SERVER_COMBOS = {
    ("VAELoader", "vae_name"),
    ("LoadImageDataSetFromFolder", "folder"),
    ("LoadImageTextDataSetFromFolder", "folder"),
    ("LoadVideoDataSetFromFolder", "folder"),
    ("LoadVideoTextDataSetFromFolder", "folder"),
    ("TrainLoraNode", "existing_lora"),
    ("SelectCLIPDevice", "device"),
    ("SelectModelDevice", "device"),
    ("SelectVAEDevice", "device"),
}
MODULE_SEGMENTS = {"3d": "three_d"}
LINK = ("Upstream_0", 0)


class Input(NamedTuple):
    name: str
    required: bool
    kind: str
    options: dict[str, Any]
    choices: list[Any]

    @property
    def has_default(self) -> bool:
        return self.kind != "link" and "default" in self.options


def _kind(node_name, name, input_type, options):
    if options.get("forceInput"):
        return "link"
    if isinstance(input_type, list) or input_type == "COMBO":
        choices = input_type if isinstance(input_type, list) else options.get("options")
        from_server = (
            not choices
            or any(options.get(flag) for flag in SERVER_COMBO_FLAGS)
            or (node_name, name) in SERVER_COMBOS
        )
        return "server_combo" if from_server else "combo"
    widget = WIDGET_KINDS.get(options.get("widgetType", input_type))
    if widget:
        return widget
    if "default" in options or options.get("socketless"):
        return "any"
    produced = any(part in OUTPUT_TYPES for part in input_type.split(","))
    if input_type not in ("*", "COMFY_MATCHTYPE_V3") and not produced:
        return "any"
    return "link"


def inputs_of(node_name):
    spec = SPECS[node_name]
    for section in ("required", "optional"):
        for name in spec["input_order"].get(section, []):
            input_type, *rest = spec["input"][section][name]
            options = rest[0] if rest else {}
            choices = input_type if isinstance(input_type, list) else []
            if input_type == "COMBO":
                choices = options.get("options", [])
            yield Input(
                name=name,
                required=section == "required",
                kind=_kind(node_name, name, input_type, options),
                options=options,
                choices=choices,
            )


def _is_dynamic(spec):
    return any(
        isinstance(input_spec[0], str) and input_spec[0] in DYNAMIC_TYPES
        for section in ("required", "optional")
        for input_spec in spec["input"].get(section, {}).values()
    )


STATIC_NODES = sorted(name for name, spec in SPECS.items() if not _is_dynamic(spec))
DYNAMIC_NODES = sorted(set(SPECS) - set(STATIC_NODES))
# Every dynamic node has a class, whose dotted inputs became plain attributes.
IMPLEMENTED_DYNAMIC_NODES = DYNAMIC_NODES


def _import_all(package):
    for info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        if info.ispkg:
            _import_all(module)


def _node_classes():
    _import_all(nodes_package)
    classes = {}
    pending = [BaseNode]
    while pending:
        for cls in pending.pop().__subclasses__():
            pending.append(cls)
            if cls.__module__.startswith(f"{nodes_package.__name__}."):
                class_type = getattr(cls, "CLASS_TYPE", cls.__name__)
                classes.setdefault(class_type, []).append(cls)
    return classes


NODE_CLASSES = _node_classes()


def node_class(node_name):
    if node_name not in NODE_CLASSES:
        pytest.fail(f"no class for the ComfyUI node {node_name!r}")
    return NODE_CLASSES[node_name][0]


def attribute_name(node_name, input_name):
    renamed = getattr(node_class(node_name), "INPUT_NAMES", {})
    return {comfy: attribute for attribute, comfy in renamed.items()}.get(
        input_name, input_name
    )


def sample_value(item):
    if item.kind == "link":
        return LINK
    if item.kind == "combo":
        return item.choices[0]
    if item.kind == "server_combo":
        return "sample.safetensors"
    if item.kind in ("int", "float"):
        low, high = item.options.get("min"), item.options.get("max")
        if low is not None:
            return low
        return 0 if high is None else min(high, 0)
    if item.kind == "string":
        return "text"
    if item.kind == "boolean":
        return False
    return "sample"


def sent(value):
    # ComfyUI reads a list as a link, so list values go wrapped.
    return {"__value__": value} if isinstance(value, list) else value


def build(node_name, values=None):
    cls = node_class(node_name)
    arguments = {
        item.name: sample_value(item)
        for item in inputs_of(node_name)
        if item.required and not item.has_default
    }
    arguments.update(values or {})
    return cls(
        node_id=f"{cls.__name__}_0",
        **{attribute_name(node_name, name): value for name, value in arguments.items()},
    )


def sent_inputs(node):
    return node.get_comfy_ui_workflow_format()[node.node_id]["inputs"]


def mentions(node_name, item):
    attribute = attribute_name(node_name, item.name)
    return f"{re.escape(attribute)}|{re.escape(attribute.replace('_', ' '))}"


def inputs_where(predicate):
    return [
        pytest.param(node_name, item, id=f"{node_name}-{item.name}")
        for node_name in STATIC_NODES
        for item in inputs_of(node_name)
        if predicate(item)
    ]


def test_every_static_node_has_exactly_one_class():
    missing = [name for name in STATIC_NODES if name not in NODE_CLASSES]
    duplicated = sorted(name for name, found in NODE_CLASSES.items() if len(found) > 1)

    assert (missing, duplicated) == ([], [])


def test_classes_exist_only_for_static_and_implemented_dynamic_nodes():
    assert sorted(set(NODE_CLASSES) - set(STATIC_NODES)) == IMPLEMENTED_DYNAMIC_NODES


def test_every_dynamic_node_has_exactly_one_class():
    missing = [name for name in DYNAMIC_NODES if name not in NODE_CLASSES]

    assert missing == []


@pytest.mark.parametrize("node_name", STATIC_NODES + IMPLEMENTED_DYNAMIC_NODES)
def test_category_matches_comfy_ui(node_name):
    assert node_class(node_name).CATEGORY == SPECS[node_name]["category"]


@pytest.mark.parametrize("node_name", STATIC_NODES + IMPLEMENTED_DYNAMIC_NODES)
def test_outputs_follow_comfy_ui_order(node_name):
    names = SPECS[node_name]["output_name"]

    assert node_class(node_name).OUTPUTS == {
        name: index for index, name in enumerate(names)
    }


@pytest.mark.parametrize("node_name", STATIC_NODES + IMPLEMENTED_DYNAMIC_NODES)
def test_module_follows_the_category(node_name):
    segments = [
        re.sub(r"[^0-9a-z]+", "_", part.lower()).strip("_")
        for part in SPECS[node_name]["category"].split("/")
    ]
    module = ".".join(
        [nodes_package.__name__, *(MODULE_SEGMENTS.get(s, s) for s in segments)]
    )

    assert node_class(node_name).__module__ == module


@pytest.mark.parametrize("node_name", STATIC_NODES)
def test_default_format_matches_comfy_ui(node_name):
    expected = {}
    for item in inputs_of(node_name):
        if item.has_default:
            expected[item.name] = sent(item.options["default"])
        elif item.required:
            expected[item.name] = sample_value(item)

    node = build(node_name)

    assert node.get_comfy_ui_workflow_format() == {
        node.node_id: {"class_type": node_name, "inputs": expected}
    }


@pytest.mark.parametrize(
    "node_name",
    [
        node_name
        for node_name in STATIC_NODES
        if any(item.kind != "link" for item in inputs_of(node_name))
    ],
)
def test_widgets_accept_links(node_name):
    widgets = [item for item in inputs_of(node_name) if item.kind != "link"]

    sent_links = {
        item.name: sent_inputs(build(node_name, {item.name: LINK}))[item.name]
        for item in widgets
    }

    assert sent_links == {item.name: LINK for item in widgets}


@pytest.mark.parametrize(
    ("node_name", "item"),
    inputs_where(lambda item: not item.required and not item.has_default),
)
def test_optional_input_without_default_is_sent_only_when_given(node_name, item):
    value = sample_value(item)

    assert item.name not in sent_inputs(build(node_name))
    assert sent_inputs(build(node_name, {item.name: value}))[item.name] == value


@pytest.mark.parametrize(
    ("node_name", "item"), inputs_where(lambda item: item.kind in ("int", "float"))
)
def test_number_respects_its_closed_range_and_rejects_bool(node_name, item):
    low, high = item.options.get("min"), item.options.get("max")
    for bound in (low, high):
        if bound is not None:
            assert sent_inputs(build(node_name, {item.name: bound}))[item.name] == bound
    outside = []
    if low is not None:
        outside.append(low - 1 if isinstance(low, int) else low - max(1.0, abs(low)))
    if high is not None:
        outside.append(
            high + 1 if isinstance(high, int) else high + max(1.0, abs(high))
        )
    for value in (*outside, True):
        with pytest.raises(InvalidNodeParameterValue, match=mentions(node_name, item)):
            build(node_name, {item.name: value})


@pytest.mark.parametrize(
    ("node_name", "item"), inputs_where(lambda item: item.kind == "float")
)
def test_float_rejects_nan_and_infinity(node_name, item):
    # The API format is JSON, which has no NaN or infinity.
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(InvalidNodeParameterValue, match=mentions(node_name, item)):
            build(node_name, {item.name: value})


@pytest.mark.parametrize(
    ("node_name", "item"), inputs_where(lambda item: item.kind == "combo")
)
def test_combo_accepts_every_option_and_rejects_others(node_name, item):
    for option in item.choices:
        assert sent_inputs(build(node_name, {item.name: option}))[item.name] == option
    with pytest.raises(InvalidNodeParameterValue, match=mentions(node_name, item)):
        build(node_name, {item.name: "__not_an_option__"})


@pytest.mark.parametrize(
    ("node_name", "item"), inputs_where(lambda item: item.kind == "server_combo")
)
def test_server_combo_accepts_any_name_and_rejects_empty(node_name, item):
    name = "my_file.safetensors"

    assert sent_inputs(build(node_name, {item.name: name}))[item.name] == name
    with pytest.raises(InvalidNodeParameterValue, match=mentions(node_name, item)):
        build(node_name, {item.name: ""})


@pytest.mark.parametrize(
    ("node_name", "item"), inputs_where(lambda item: item.kind == "boolean")
)
def test_boolean_rejects_values_that_are_not_bool(node_name, item):
    assert sent_inputs(build(node_name, {item.name: True}))[item.name] is True
    with pytest.raises(InvalidNodeParameterValue, match=mentions(node_name, item)):
        build(node_name, {item.name: 1})
