import inspect
import json
from pathlib import Path

import pytest

from comfy_ui_api_workflow._impl.nodes.base import BaseNode
from comfy_ui_api_workflow._impl.nodes.image.filters import ImageBlend
from comfy_ui_api_workflow._impl.nodes.image.upscaling import ImageScale
from comfy_ui_api_workflow._impl.nodes.model.loaders import CheckpointLoaderSimple
from comfy_ui_api_workflow._impl.nodes.model.sampling import KSampler
from comfy_ui_api_workflow._impl.nodes.video import CreateVideo
from comfy_ui_api_workflow._impl.workflow import _node_types

OBJECT_INFO = Path(__file__).resolve().parent.parent / "new_object_info.json"
NODE_TYPES = sorted(_node_types().values(), key=lambda node_type: node_type.CLASS_TYPE)


def input_names(node_type):
    return {name for name in inspect.signature(node_type).parameters} - {"node_id"}


def socket_type(entry):
    return entry[0] if isinstance(entry[0], str) else "V1 COMBO"


def flattened_inputs(spec):
    inputs = {}

    def add(name, entry):
        options = entry[1] if len(entry) > 1 else {}
        if entry[0] == "COMFY_AUTOGROW_V3":
            ((_, value),) = options["template"]["input"]["required"].items()
            for letter in options["template"]["names"]:
                inputs[f"{name}.{letter}"] = socket_type(value)
            return
        inputs[name] = socket_type(entry)
        if entry[0] == "COMFY_DYNAMICCOMBO_V3":
            for option in options["options"]:
                for section in ("required", "optional"):
                    for sub, sub_entry in option["inputs"].get(section, {}).items():
                        add(f"{name}.{sub}", sub_entry)

    for section in ("required", "optional"):
        for name, entry in spec["input"].get(section, {}).items():
            add(name, entry)
    return inputs


class UntypedNode(BaseNode):
    __slots__ = ("model",)
    CATEGORY = "test"

    def __init__(self, node_id, model):
        super().__init__(node_id=node_id)
        self.model = model


def test_nodes_without_socket_types_declare_none():
    assert (UntypedNode.RETURN_TYPES, UntypedNode.INPUT_SOCKET_TYPES) == ((), {})


def test_every_output_of_a_library_node_has_a_type():
    missing = [
        node_type.CLASS_TYPE
        for node_type in NODE_TYPES
        if len(node_type.RETURN_TYPES) != len(node_type.OUTPUTS)
        or not all(isinstance(kind, str) and kind for kind in node_type.RETURN_TYPES)
    ]

    assert missing == []


def test_every_input_of_a_library_node_has_a_socket_type():
    missing = [
        node_type.CLASS_TYPE
        for node_type in NODE_TYPES
        if set(node_type.INPUT_SOCKET_TYPES) != input_names(node_type)
        or not all(
            isinstance(kind, str) and kind
            for kind in node_type.INPUT_SOCKET_TYPES.values()
        )
    ]

    assert missing == []


@pytest.mark.parametrize(
    ("node_type", "name", "kind"),
    [
        (KSampler, "sampler_name", "V1 COMBO"),
        (KSampler, "scheduler", "V1 COMBO"),
        (CheckpointLoaderSimple, "ckpt_name", "V1 COMBO"),
        (ImageScale, "upscale_method", "V1 COMBO"),
        (CreateVideo, "bit_depth", "COMBO"),
        (ImageBlend, "blend_mode", "COMBO"),
    ],
)
def test_v1_combos_have_their_own_socket_type(node_type, name, kind):
    assert node_type.INPUT_SOCKET_TYPES[name] == kind


@pytest.mark.skipif(not OBJECT_INFO.exists(), reason=f"{OBJECT_INFO.name} not found")
def test_socket_types_match_comfy_ui():
    specs = json.loads(OBJECT_INFO.read_text(encoding="utf-8"))
    different = []
    for node_type in NODE_TYPES:
        spec = specs[node_type.CLASS_TYPE]
        attributes = {comfy: name for name, comfy in node_type.INPUT_NAMES.items()}
        inputs = {
            attributes.get(name, name): kind
            for name, kind in flattened_inputs(spec).items()
        }
        if node_type.RETURN_TYPES != tuple(spec["output"]):
            different.append((node_type.CLASS_TYPE, "outputs"))
        if node_type.INPUT_SOCKET_TYPES != inputs:
            different.append((node_type.CLASS_TYPE, "inputs"))

    assert different == []
