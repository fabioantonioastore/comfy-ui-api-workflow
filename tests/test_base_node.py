from typing import Any

import pytest

from comfy_ui_api_workflow._impl.errors.node import InvalidNodeParameterValue
from comfy_ui_api_workflow._impl.errors.output import OutputNotFoundError
from comfy_ui_api_workflow._impl.nodes.base import BaseNode


class TextNode(BaseNode):
    __slots__ = ("text", "clip")
    CATEGORY = "test"
    OUTPUTS = {"CONDITIONING": 0}

    def __init__(self, node_id, text, clip):
        super().__init__(node_id=node_id)
        self.text = text
        self.clip = clip


class PartialNode(BaseNode):
    __slots__ = ("set_input", "unset_input")
    CATEGORY = "test"

    def __init__(self, node_id):
        super().__init__(node_id=node_id)
        self.set_input = 1


class IntermediateNode(BaseNode):
    __slots__ = ("width",)

    def __init__(self, node_id, width):
        super().__init__(node_id=node_id)
        self.width = width


class LeafNode(IntermediateNode):
    __slots__ = ("height",)
    CATEGORY = "test"

    def __init__(self, node_id, width, height):
        super().__init__(node_id=node_id, width=width)
        self.height = height


class RenamedNode(BaseNode):
    __slots__ = ("model",)
    CATEGORY = "test"
    CLASS_TYPE = "Renamed Node"

    def __init__(self, node_id, model):
        super().__init__(node_id=node_id)
        self.model = model


class DottedInputNode(BaseNode):
    __slots__ = ("time_embed", "output_")
    CATEGORY = "test"
    INPUT_NAMES = {"time_embed": "time_embed.", "output_": "output"}

    def __init__(self, node_id, time_embed, output_):
        super().__init__(node_id=node_id)
        self.time_embed = time_embed
        self.output_ = output_


class ListWidgetNode(BaseNode):
    __slots__ = ("editor_state", "image")
    CATEGORY = "test"

    def __init__(self, node_id, editor_state, image):
        super().__init__(node_id=node_id)
        self.editor_state = editor_state
        self.image = image


class TypedNode(BaseNode):
    __slots__ = ("steps", "model", "mask")
    CATEGORY = "test"
    OUTPUTS = {"LATENT": 0}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        steps: int | tuple[str, int] = 20,
        mask: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.steps = steps
        if mask is not None:
            self.mask = mask


class RangeNode(BaseNode):
    __slots__ = ("low", "high")
    CATEGORY = "test"

    def __init__(self, node_id: str, low: int = 0, high: int = 1) -> None:
        super().__init__(node_id=node_id)
        self.low = low
        self.high = high
        if self.low > self.high:
            raise InvalidNodeParameterValue("low must not be above high")


class ShiftedRangeNode(RangeNode):
    def __init__(self, node_id: str, low: int = 5, high: int = 6) -> None:
        super().__init__(node_id=node_id)
        self.low = low
        self.high = high


class PlainRangeNode(RangeNode):
    pass


class MutableDefaultNode(BaseNode):
    __slots__ = ("state", "note")
    CATEGORY = "test"

    def __init__(
        self, node_id: str, state: Any = None, note: tuple[str, int] | None = None
    ) -> None:
        super().__init__(node_id=node_id)
        self.state = {} if state is None else state
        if note is not None:
            self.note = note


class TemporaryInputNode(BaseNode):
    __slots__ = ("mask",)
    CATEGORY = "test"

    def __init__(self, node_id: str, mask: tuple[str, int] | None = None) -> None:
        super().__init__(node_id=node_id)
        self.mask = ("Placeholder_0", 0)
        del self.mask
        if mask is not None:
            self.mask = mask


def test_comfy_ui_format_is_keyed_by_node_id_with_slots_as_inputs():
    node = TextNode(node_id="TextNode_0", text="a cat", clip=("Loader_0", 1))

    assert node.get_comfy_ui_workflow_format() == {
        "TextNode_0": {
            "class_type": "TextNode",
            "inputs": {"text": "a cat", "clip": ("Loader_0", 1)},
        }
    }


def test_unset_slots_are_left_out_of_the_inputs():
    node = PartialNode(node_id="PartialNode_0")

    assert node.get_comfy_ui_workflow_format()["PartialNode_0"]["inputs"] == {
        "set_input": 1
    }


def test_inputs_from_intermediate_classes_are_included():
    node = LeafNode(node_id="LeafNode_0", width=512, height=768)

    assert node.get_comfy_ui_workflow_format()["LeafNode_0"]["inputs"] == {
        "width": 512,
        "height": 768,
    }


def test_output_links_come_from_the_class_outputs():
    node = TextNode(node_id="TextNode_0", text="a cat", clip=("Loader_0", 1))

    assert node.output["CONDITIONING"] == ("TextNode_0", 0)


def test_node_without_outputs_has_no_output_to_link():
    node = PartialNode(node_id="PartialNode_0")

    with pytest.raises(OutputNotFoundError):
        node.output["LATENT"]


def test_nodes_reject_undeclared_attributes():
    node = TextNode(node_id="TextNode_0", text="a cat", clip=("Loader_0", 1))

    with pytest.raises(AttributeError):
        node.txet = "typo"


def test_class_type_defaults_to_the_class_name():
    assert TextNode.CLASS_TYPE == "TextNode"
    assert IntermediateNode.CLASS_TYPE == "IntermediateNode"
    assert LeafNode.CLASS_TYPE == "LeafNode"


def test_explicit_class_type_is_sent_as_the_class_type():
    node = RenamedNode(node_id="RenamedNode_0", model=("Loader_0", 0))

    assert (
        node.get_comfy_ui_workflow_format()["RenamedNode_0"]["class_type"]
        == "Renamed Node"
    )


def test_input_names_map_attributes_to_comfy_ui_input_names():
    node = DottedInputNode(node_id="DottedInputNode_0", time_embed=0.5, output_="depth")

    assert node.get_comfy_ui_workflow_format()["DottedInputNode_0"]["inputs"] == {
        "time_embed.": 0.5,
        "output": "depth",
    }


def test_list_values_are_wrapped_so_comfy_ui_does_not_read_them_as_links():
    node = ListWidgetNode(
        node_id="ListWidgetNode_0", editor_state=[{"x": 0}], image=["LoadImage_0", 0]
    )

    assert node.get_comfy_ui_workflow_format()["ListWidgetNode_0"]["inputs"] == {
        "editor_state": {"__value__": [{"x": 0}]},
        "image": ["LoadImage_0", 0],
    }


def test_empty_list_value_is_wrapped():
    node = ListWidgetNode(
        node_id="ListWidgetNode_0", editor_state=[], image=("LoadImage_0", 0)
    )

    assert node.get_comfy_ui_workflow_format()["ListWidgetNode_0"]["inputs"][
        "editor_state"
    ] == {"__value__": []}


def test_optional_input_of_a_node_outside_a_workflow_can_be_deleted():
    node = TypedNode(
        node_id="TypedNode_0", model=("Loader_0", 0), mask=("LoadImage_1", 1)
    )

    del node.mask

    assert node.inputs() == {"steps": 20, "model": ("Loader_0", 0)}


@pytest.mark.parametrize("name", ["model", "steps"], ids=["required", "with-default"])
def test_input_that_is_not_optional_cannot_be_deleted(name):
    node = TypedNode(node_id="TypedNode_0", model=("Loader_0", 0))
    before = node.inputs()

    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        delattr(node, name)

    assert str(excinfo.value) == f"{name} cannot be removed because it is not optional"
    assert node.inputs() == before


def test_typed_inputs_accept_values_and_links_of_their_type():
    node = TypedNode(
        node_id="TypedNode_0",
        model=("Loader_0", 0),
        steps=("PrimitiveInt_1", 0),
        mask=("LoadImage_2", 1),
    )

    assert node.inputs() == {
        "steps": ("PrimitiveInt_1", 0),
        "model": ("Loader_0", 0),
        "mask": ("LoadImage_2", 1),
    }


def test_typed_input_rejects_a_value_of_another_type():
    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        TypedNode(node_id="TypedNode_0", model=("Loader_0", 0), steps="20")

    assert str(excinfo.value) == "steps must be an int or a link, not str"


def test_link_only_input_rejects_a_value():
    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        TypedNode(node_id="TypedNode_0", model="model.safetensors")

    assert str(excinfo.value) == "model must be a link, not str"


def test_assigning_a_value_of_another_type_keeps_the_previous_value():
    node = TypedNode(node_id="TypedNode_0", model=("Loader_0", 0))

    with pytest.raises(InvalidNodeParameterValue):
        node.steps = 2.5

    assert node.steps == 20


def test_assigning_a_value_of_another_type_leaves_an_unset_input_unset():
    node = TypedNode(node_id="TypedNode_0", model=("Loader_0", 0))

    with pytest.raises(InvalidNodeParameterValue):
        node.mask = "mask.png"

    assert "mask" not in node.inputs()


def test_node_id_must_be_a_str():
    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        TextNode(node_id=7, text="a cat", clip=("Loader_0", 1))

    assert str(excinfo.value) == "node_id must be a str, not int"


def test_inputs_without_annotations_accept_any_value():
    node = TextNode(node_id="TextNode_0", text=5, clip=None)

    assert node.inputs() == {"text": 5, "clip": None}


def test_change_that_breaks_the_node_validation_keeps_the_previous_value():
    node = RangeNode(node_id="RangeNode_0", low=0, high=1)

    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        node.low = 2

    assert str(excinfo.value) == "low must not be above high"
    assert (node.low, node.high) == (0, 1)


class RequiredRangeNode(BaseNode):
    __slots__ = ("low", "high")
    CATEGORY = "test"

    def __init__(self, node_id: str, low: int, high: int) -> None:
        super().__init__(node_id=node_id)
        self.low = low
        self.high = high
        if self.low > self.high:
            raise InvalidNodeParameterValue("low must not be above high")


def test_change_to_a_node_whose_inputs_are_all_required_is_validated():
    node = RequiredRangeNode(node_id="RequiredRangeNode_0", low=0, high=1)

    with pytest.raises(InvalidNodeParameterValue, match="low must not be above high"):
        node.low = 2

    assert (node.low, node.high) == (0, 1)


def test_valid_change_is_kept():
    node = RangeNode(node_id="RangeNode_0", low=0, high=1)

    node.high = 10
    node.low = 5

    assert (node.low, node.high) == (5, 10)


def test_subclass_without_its_own_init_validates_changes():
    node = PlainRangeNode(node_id="PlainRangeNode_0")

    with pytest.raises(InvalidNodeParameterValue):
        node.low = 2

    assert node.low == 0


def test_subclass_init_is_not_validated_before_it_finishes():
    node = ShiftedRangeNode(node_id="ShiftedRangeNode_0")

    assert (node.low, node.high) == (5, 6)


def test_node_whose_constructor_does_not_take_its_inputs_is_not_validated_again():
    node = PartialNode(node_id="PartialNode_0")

    node.unset_input = 5

    assert node.inputs() == {"set_input": 1, "unset_input": 5}


class UnrebuiltConditioningNode(BaseNode):
    __slots__ = ("positive", "negative", "strength")
    CATEGORY = "test"

    def __init__(self, node_id, positive, negative):
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.strength = 1.0


def test_node_that_is_not_rebuilt_rejects_the_same_link_for_positive_and_negative():
    node = UnrebuiltConditioningNode(
        node_id="UnrebuiltConditioningNode_0",
        positive=("Text_1", 0),
        negative=("Text_2", 0),
    )

    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        node.negative = ("Text_1", 0)

    assert str(excinfo.value) == "positive and negative must not use the same input"
    assert node.negative == ("Text_2", 0)


def test_input_deleted_during_construction_is_not_validated():
    node = TemporaryInputNode(node_id="TemporaryInputNode_0")

    assert node.inputs() == {}


def test_input_whose_constructor_always_fills_it_cannot_be_deleted():
    node = MutableDefaultNode(node_id="MutableDefaultNode_0", state={"zoom": 2})

    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        del node.state

    assert str(excinfo.value) == "state cannot be removed because it is not optional"
    assert node.state == {"zoom": 2}


def test_input_whose_constructor_leaves_it_out_can_be_deleted():
    node = MutableDefaultNode(node_id="MutableDefaultNode_0", note=("Text_1", 0))

    del node.note

    assert node.inputs() == {"state": {}}
