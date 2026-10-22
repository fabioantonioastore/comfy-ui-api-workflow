import json
from pathlib import Path

import pytest

from comfy_ui_api_workflow._impl.errors.node import InvalidNodeParameterValue
from comfy_ui_api_workflow._impl.nodes.image.filters import ColorTransfer
from comfy_ui_api_workflow._impl.nodes.model.loaders import LoadLoraModel
from comfy_ui_api_workflow._impl.nodes.utilities.looping import StartLoop
from comfy_ui_api_workflow._impl.workflow import _load_node, _node_types

OBJECT_INFO = Path(__file__).resolve().parent.parent / "new_object_info.json"

if not OBJECT_INFO.exists():
    pytest.skip(
        f"{OBJECT_INFO.name} not found at the project root", allow_module_level=True
    )

SPECS = json.loads(OBJECT_INFO.read_text(encoding="utf-8"))
AUTOGROW = "COMFY_AUTOGROW_V3"
COMBO = "COMFY_DYNAMICCOMBO_V3"
GROUP = "COMFY_DYNAMICGROUP_V3"
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
LINK = ("Upstream_0", 0)
# Covered on their own in test_dynamic_nodes.py.
HAND_WRITTEN = {"ComfyMathExpression", "SaveVideo"}
NODES = sorted(
    name
    for name, spec in SPECS.items()
    if name not in HAND_WRITTEN
    and any(
        isinstance(entry[0], str) and entry[0] in (AUTOGROW, COMBO, GROUP)
        for section in ("required", "optional")
        for entry in spec["input"].get(section, {}).values()
    )
)
NODE_TYPES = _node_types()


def options_of(entry):
    return entry[1] if len(entry) > 1 else {}


def kind(entry):
    input_type, options = entry[0], options_of(entry)
    if options.get("forceInput"):
        return "link"
    if isinstance(input_type, list) or input_type == "COMBO":
        choices = input_type if isinstance(input_type, list) else options.get("options")
        return "combo" if choices else "server_combo"
    widget = WIDGET_KINDS.get(input_type)
    if widget:
        return widget
    if "default" in options or options.get("socketless"):
        return "any"
    produced = any(part in OUTPUT_TYPES for part in input_type.split(","))
    if input_type not in ("*", "COMFY_MATCHTYPE_V3") and not produced:
        return "any"
    return "link"


def choices_of(entry):
    if isinstance(entry[0], list):
        return entry[0]
    return options_of(entry).get("options", [])


# Each input gets its own link, since positive and negative cannot share one.
def link_for(name):
    return (f"Upstream_{name}", 0)


def sample(entry, path):
    found = kind(entry)
    options = options_of(entry)
    if found == "link":
        return link_for(path)
    if found == "combo":
        return choices_of(entry)[0]
    if found == "server_combo":
        return "sample.safetensors"
    if found in ("int", "float"):
        low, high = options.get("min"), options.get("max")
        if low is not None:
            return low
        return 0 if high is None else min(high, 0)
    if found == "string":
        return "text"
    if found == "boolean":
        return False
    return "sample"


def has_default(entry):
    return kind(entry) != "link" and "default" in options_of(entry)


def sent(value):
    return {"__value__": value} if isinstance(value, list) else value


def attribute(path):
    return path.replace(".", "_")


def ordered(inputs):
    for section in ("required", "optional"):
        for name, entry in inputs.get(section, {}).items():
            yield name, entry, section == "required"


def top_inputs(node_name):
    spec = SPECS[node_name]
    for section in ("required", "optional"):
        for name in spec["input_order"].get(section, []):
            yield name, spec["input"][section][name], section == "required"


class Expected:
    """What ComfyUI gets for a node, worked out from object_info alone."""

    def __init__(self, node_name, choices=None, given=None):
        self.choices = choices or {}
        self.given = given or {}
        self.sent = {}
        self.needed = {}
        for name, entry, required in top_inputs(node_name):
            self.add(name, entry, required)

    def add(self, path, entry, required):
        input_type, options = entry[0], options_of(entry)
        if input_type == COMBO:
            if path in self.choices or required:
                key = self.choices.get(path, options["options"][0]["key"])
                self.sent[path] = key
                option = next(o for o in options["options"] if o["key"] == key)
                for sub, sub_entry, sub_required in ordered(option["inputs"]):
                    self.add(f"{path}.{sub}", sub_entry, sub_required)
        elif input_type == AUTOGROW:
            template = options["template"]
            names = template.get("names") or [
                f"{template['prefix']}{index}" for index in range(template["max"])
            ]
            for index, name in enumerate(names):
                slot = f"{path}.{name}"
                if slot in self.given:
                    self.sent[slot] = self.given[slot]
                elif required and index < template["min"]:
                    self.sent[slot] = self.needed[slot] = LINK
        elif input_type == GROUP:
            rows = {
                int(key.split(".")[1])
                for key in self.given
                if key.startswith(f"{path}.")
            }
            if required and len(rows) < options["min"]:
                rows |= set(range(options["min"]))
            for row in sorted(rows):
                for field, field_entry, field_required in ordered(options["template"]):
                    self.add(f"{path}.{row}.{field}", field_entry, field_required)
        elif path in self.given:
            self.sent[path] = sent(self.given[path])
        elif has_default(entry):
            self.sent[path] = sent(options["default"])
        elif required:
            self.sent[path] = self.needed[path] = sample(entry, path)

    def build(self, node_name):
        values = {**self.choices, **self.needed, **self.given}
        return NODE_TYPES[node_name](
            node_id=f"{node_name}_0",
            **{attribute(path): value for path, value in values.items()},
        )

    def workflow_format(self, node_name):
        return {f"{node_name}_0": {"class_type": node_name, "inputs": dict(self.sent)}}


def branch_choices(path, entry, chosen):
    for option in options_of(entry)["options"]:
        choices = {**chosen, path: option["key"]}
        yield option, choices
        for sub, sub_entry, _ in ordered(option["inputs"]):
            if sub_entry[0] == COMBO:
                yield from branch_choices(f"{path}.{sub}", sub_entry, choices)


def branches():
    for node_name in NODES:
        for name, entry, _ in top_inputs(node_name):
            if entry[0] != COMBO:
                continue
            for option, choices in branch_choices(name, entry, {}):
                label = "/".join(f"{key}={value}" for key, value in choices.items())
                yield pytest.param(
                    node_name, choices, option, id=f"{node_name}-{label}"
                )


BRANCHES = list(branches())


def inputs_of_option(path, option):
    return {f"{path}.{sub}": entry for sub, entry, _ in ordered(option["inputs"])}


def combo_entry(node_name, path):
    first, *rest = path.split(".")
    entry = {name: entry for name, entry, _ in top_inputs(node_name)}[first]
    walked = first
    for name in rest:
        inputs = {}
        for option in options_of(entry)["options"]:
            inputs.update(inputs_of_option(walked, option))
        walked = f"{walked}.{name}"
        entry = inputs[walked]
    return entry


@pytest.mark.parametrize("node_name", NODES)
def test_dotted_inputs_are_attributes_with_underscores(node_name):
    renamed = NODE_TYPES[node_name].INPUT_NAMES

    assert renamed
    assert all(name == attribute(comfy) for name, comfy in renamed.items())


@pytest.mark.parametrize("node_name", NODES)
def test_default_format_matches_comfy_ui(node_name):
    expected = Expected(node_name)

    node = expected.build(node_name)

    assert node.get_comfy_ui_workflow_format() == expected.workflow_format(node_name)


@pytest.mark.parametrize(("node_name", "choices", "option"), BRANCHES)
def test_each_branch_sends_its_inputs_with_their_defaults(node_name, choices, option):
    expected = Expected(node_name, choices)

    node = expected.build(node_name)
    workflow_format = node.get_comfy_ui_workflow_format()
    data = json.loads(json.dumps(workflow_format))[node.node_id]

    assert workflow_format == expected.workflow_format(node_name)
    assert _load_node(node.node_id, data).get_comfy_ui_workflow_format() == (
        workflow_format
    )


@pytest.mark.parametrize(("node_name", "choices", "option"), BRANCHES)
def test_inputs_of_other_branches_are_rejected(node_name, choices, option):
    path = list(choices)[-1]
    mine = inputs_of_option(path, option)
    others = {
        sub: entry
        for other in options_of(combo_entry(node_name, path))["options"]
        for sub, entry in inputs_of_option(path, other).items()
        if sub not in mine
    }
    if not others:
        pytest.skip("every option of this combo opens the same inputs")
    for sub, entry in others.items():
        value = (
            options_of(entry)["options"][0]["key"]
            if entry[0] == COMBO
            else sample(entry, sub)
        )
        expected = Expected(node_name, choices, {sub: value})

        with pytest.raises(InvalidNodeParameterValue, match=f"{attribute(sub)} only"):
            expected.build(node_name)


@pytest.mark.parametrize(("node_name", "choices", "option"), BRANCHES)
def test_branch_widgets_are_checked_against_their_branch(node_name, choices, option):
    path = list(choices)[-1]
    checked = 0
    for sub, entry in inputs_of_option(path, option).items():
        found, options = kind(entry), options_of(entry)
        wrong = []
        if found in ("int", "float") and options.get("max") is not None:
            wrong.append(options["max"] + 1)
        if found in ("int", "float") and options.get("min") is not None:
            wrong.append(options["min"] - 1)
        if found == "combo":
            wrong.append("__not_an_option__")
        if found == "boolean":
            wrong.append(1)
        for value in wrong:
            with pytest.raises(InvalidNodeParameterValue, match=attribute(sub)):
                Expected(node_name, choices, {sub: value}).build(node_name)
            checked += 1
        if found not in ("link", "any") and entry[0] != COMBO:
            linked = Expected(node_name, choices, {sub: LINK})
            assert linked.build(node_name).get_comfy_ui_workflow_format() == (
                linked.workflow_format(node_name)
            )
    if not checked:
        pytest.skip("no bounded widget in this branch")


AUTOGROWS = [
    pytest.param(node_name, name, entry, id=f"{node_name}-{name}")
    for node_name in NODES
    for name, entry, _ in top_inputs(node_name)
    if entry[0] == AUTOGROW
]


@pytest.mark.parametrize(("node_name", "name", "entry"), AUTOGROWS)
def test_every_autogrow_slot_is_sent_with_its_dotted_name(node_name, name, entry):
    template = options_of(entry)["template"]
    names = template.get("names") or [
        f"{template['prefix']}{index}" for index in range(template["max"])
    ]
    given = {
        f"{name}.{slot}": (f"Upstream_{index}", 0) for index, slot in enumerate(names)
    }
    expected = Expected(node_name, given=given)

    node = expected.build(node_name)

    assert node.get_comfy_ui_workflow_format() == expected.workflow_format(node_name)


def test_switching_a_branch_leaves_no_input_of_the_old_one():
    node = ColorTransfer(
        node_id="1",
        image_target=LINK,
        image_ref=("Upstream_1", 0),
        method="mkl_lab",
        source_stats="target_frame",
        source_stats_target_index=12,
    )

    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        node.source_stats = "uniform"
    del node.source_stats_target_index
    node.source_stats = "uniform"

    assert str(excinfo.value) == (
        "source_stats_target_index only applies when source_stats is 'target_frame'"
    )
    assert node.get_comfy_ui_workflow_format()[node.node_id]["inputs"] == {
        "image_target": LINK,
        "image_ref": ("Upstream_1", 0),
        "method": "mkl_lab",
        "source_stats": "uniform",
        "strength": 1.0,
    }


def test_branch_default_is_sent_but_not_stored():
    node = ColorTransfer(
        node_id="1",
        image_target=LINK,
        image_ref=LINK,
        method="histogram",
        source_stats="target_frame",
    )
    before = node.get_comfy_ui_workflow_format()["1"]["inputs"]

    node.source_stats_target_index = 7

    assert before["source_stats.target_index"] == 0
    assert (
        "source_stats_target_index"
        not in ColorTransfer(
            node_id="1", image_target=LINK, image_ref=LINK, method="histogram"
        ).inputs()
    )
    assert (
        node.get_comfy_ui_workflow_format()["1"]["inputs"]["source_stats.target_index"]
        == 7
    )


def test_branch_input_without_default_is_required_in_its_branch():
    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        StartLoop(node_id="1", mode="List")

    assert str(excinfo.value) == "mode_list is required when mode is 'List'"


def test_lora_rows_are_sent_with_the_defaults_of_each_used_row():
    node = LoadLoraModel(
        node_id="1",
        model=LINK,
        loras_0_lora_name="style.safetensors",
        loras_2_lora_name="detail.safetensors",
        loras_2_strength=0.5,
    )

    assert node.get_comfy_ui_workflow_format()["1"]["inputs"] == {
        "model": LINK,
        "loras.0.lora_name": "style.safetensors",
        "loras.0.strength": 1.0,
        "loras.0.enabled": True,
        "loras.2.lora_name": "detail.safetensors",
        "loras.2.strength": 0.5,
        "loras.2.enabled": True,
    }


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        pytest.param(
            {"loras_1_strength": 0.5},
            "loras_1_lora_name is required in row 1 of loras",
            id="row-without-a-lora",
        ),
        pytest.param(
            {"loras_0_strength": 101.0},
            "loras_0_strength must be in closed range: [-100.0, 100.0]",
            id="strength-out-of-range",
        ),
        pytest.param(
            {"loras_0_lora_name": ""},
            "loras_0_lora_name must not be empty",
            id="empty-lora-name",
        ),
    ],
)
def test_lora_rows_are_checked(inputs, message):
    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        LoadLoraModel(
            node_id="1", model=LINK, **{"loras_0_lora_name": "a.safetensors", **inputs}
        )

    assert str(excinfo.value) == message
