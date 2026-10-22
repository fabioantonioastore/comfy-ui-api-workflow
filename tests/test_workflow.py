import copy
import json
import pickle
from collections import namedtuple

import pytest

from comfy_ui_api_workflow._impl.errors.data_structure import CycleError
from comfy_ui_api_workflow._impl.errors.node import (
    InvalidLinkError,
    InvalidNodeChangeError,
    InvalidNodeParameterValue,
    NodeHasDependentsError,
    NodeNotFoundError,
)
from comfy_ui_api_workflow._impl.exceptions import NodeException
from comfy_ui_api_workflow._impl.nodes.base import BaseNode
from comfy_ui_api_workflow._impl.nodes.model.conditioning import CLIPTextEncode
from comfy_ui_api_workflow._impl.nodes.model.latent import EmptyLatentImage, VAEDecode
from comfy_ui_api_workflow._impl.nodes.model.latent.ltxv import LTXVEmptyLatentAudio
from comfy_ui_api_workflow._impl.nodes.model.loaders import CheckpointLoaderSimple
from comfy_ui_api_workflow._impl.nodes.model.sampling import KSampler
from comfy_ui_api_workflow._impl.nodes.video import CreateVideo, GetVideoComponents
from comfy_ui_api_workflow._impl.nodes.image import EmptyImage
from comfy_ui_api_workflow._impl.nodes.utilities import GetItemFromList, PreviewAny
from comfy_ui_api_workflow._impl.nodes.utilities.logic import ComfySwitchNode
from comfy_ui_api_workflow._impl.nodes.utilities.primitive import (
    PrimitiveFloat,
    PrimitiveInt,
    PrimitiveString,
)
from comfy_ui_api_workflow._impl.workflow import Workflow


class TextNode(BaseNode):
    __slots__ = ("text", "clip")
    CATEGORY = "test"
    OUTPUTS = {"CONDITIONING": 0}

    def __init__(self, node_id, text, clip):
        super().__init__(node_id=node_id)
        self.text = text
        self.clip = clip


class LatentNode(BaseNode):
    __slots__ = ("width", "height")
    CATEGORY = "test"
    OUTPUTS = {"LATENT": 0}

    def __init__(self, node_id, width=512, height=512):
        super().__init__(node_id=node_id)
        self.width = width
        self.height = height


class MaskNode(BaseNode):
    __slots__ = ("mask", "image")
    CATEGORY = "test"

    def __init__(self, node_id):
        super().__init__(node_id=node_id)


Sources = namedtuple("Sources", ["checkpoint", "positive", "negative", "latent"])


def add_sources(workflow):
    checkpoint = workflow.add_node(
        CheckpointLoaderSimple, ckpt_name="model.safetensors"
    )
    return Sources(
        checkpoint=checkpoint,
        positive=workflow.add_node(
            TextNode, text="a cat", clip=checkpoint.output["CLIP"]
        ),
        negative=workflow.add_node(
            TextNode, text="blurry", clip=checkpoint.output["CLIP"]
        ),
        latent=workflow.add_node(LatentNode),
    )


def sampler_inputs(sources, **overrides):
    return {
        "model": sources.checkpoint.output["MODEL"],
        "sampler_name": "euler",
        "scheduler": "normal",
        "positive": sources.positive.output["CONDITIONING"],
        "negative": sources.negative.output["CONDITIONING"],
        "latent_image": sources.latent.output["LATENT"],
        **overrides,
    }


def state(workflow):
    return (
        dict(workflow.nodes),
        workflow.get_comfy_ui_workflow_format(),
        {
            node_id: (workflow.dependencies(node_id), workflow.dependents(node_id))
            for node_id in workflow.nodes
        },
    )


def test_new_workflow_is_empty():
    workflow = Workflow()

    assert workflow.nodes == {}
    assert workflow.get_comfy_ui_workflow_format() == {}


def test_add_node_returns_the_created_node():
    workflow = Workflow()
    sources = add_sources(workflow)

    node = workflow.add_node(KSampler, **sampler_inputs(sources))

    assert isinstance(node, KSampler)
    assert workflow.nodes[node.node_id] is node


def test_node_ids_are_class_name_with_shared_index():
    workflow = Workflow()
    sources = add_sources(workflow)
    sampler = workflow.add_node(KSampler, **sampler_inputs(sources))

    assert [node.node_id for node in (*sources, sampler)] == [
        "CheckpointLoaderSimple_0",
        "TextNode_1",
        "TextNode_2",
        "LatentNode_3",
        "KSampler_4",
    ]


def test_workflow_format_merges_every_node():
    workflow = Workflow()
    sources = add_sources(workflow)
    sampler = workflow.add_node(KSampler, **sampler_inputs(sources))

    assert workflow.get_comfy_ui_workflow_format() == {
        node_id: node_format
        for node in (*sources, sampler)
        for node_id, node_format in node.get_comfy_ui_workflow_format().items()
    }


def test_workflow_format_reflects_changes_made_after_adding():
    workflow = Workflow()
    sampler = workflow.add_node(KSampler, **sampler_inputs(add_sources(workflow)))

    sampler.steps = 30

    assert (
        workflow.get_comfy_ui_workflow_format()[sampler.node_id]["inputs"]["steps"]
        == 30
    )


def test_nodes_are_chained_through_named_outputs():
    workflow = Workflow()
    sources = add_sources(workflow)
    first = workflow.add_node(KSampler, **sampler_inputs(sources))
    second = workflow.add_node(
        KSampler,
        **sampler_inputs(sources, latent_image=first.output["LATENT"], denoise=0.5),
    )

    sent = json.loads(json.dumps(workflow.get_comfy_ui_workflow_format()))

    assert sent[first.node_id]["inputs"]["model"] == ["CheckpointLoaderSimple_0", 0]
    assert sent[second.node_id]["inputs"]["latent_image"] == ["KSampler_4", 0]


def test_get_node_output_returns_the_node_output():
    workflow = Workflow()
    node = workflow.add_node(KSampler, **sampler_inputs(add_sources(workflow)))

    assert workflow.get_node_output(node.node_id)["LATENT"] == (node.node_id, 0)


def test_get_node_output_for_unknown_node_raises_node_not_found():
    workflow = Workflow()

    with pytest.raises(NodeNotFoundError, match="Missing_0"):
        workflow.get_node_output("Missing_0")


def test_node_not_found_is_node_exception():
    assert issubclass(NodeNotFoundError, NodeException)


def test_node_without_connections_has_no_dependencies():
    workflow = Workflow()
    checkpoint = workflow.add_node(
        CheckpointLoaderSimple, ckpt_name="model.safetensors"
    )

    assert workflow.dependencies(checkpoint.node_id) == set()


def test_node_depends_on_every_node_whose_output_it_uses():
    workflow = Workflow()
    sources = add_sources(workflow)
    sampler = workflow.add_node(KSampler, **sampler_inputs(sources))

    assert workflow.dependencies(sampler.node_id) == {node.node_id for node in sources}


def test_dependents_are_the_nodes_that_use_its_outputs():
    workflow = Workflow()
    sources = add_sources(workflow)
    sampler = workflow.add_node(KSampler, **sampler_inputs(sources))

    assert workflow.dependents(sources.checkpoint.node_id) == {
        sources.positive.node_id,
        sources.negative.node_id,
        sampler.node_id,
    }


def test_connection_written_as_a_list_is_a_dependency():
    workflow = Workflow()
    checkpoint = workflow.add_node(
        CheckpointLoaderSimple, ckpt_name="model.safetensors"
    )

    text = workflow.add_node(TextNode, text="a cat", clip=[checkpoint.node_id, 1])

    assert workflow.dependencies(text.node_id) == {checkpoint.node_id}


def test_connection_to_a_node_outside_the_workflow_is_rejected():
    workflow = Workflow()
    sources = add_sources(workflow)

    with pytest.raises(NodeNotFoundError, match="LatentNode_9"):
        workflow.add_node(
            KSampler, **sampler_inputs(sources, latent_image=("LatentNode_9", 0))
        )


def test_node_using_its_own_output_is_rejected_as_a_cycle():
    workflow = Workflow()
    sources = add_sources(workflow)

    with pytest.raises(CycleError) as excinfo:
        workflow.add_node(
            KSampler, **sampler_inputs(sources, latent_image=("KSampler_4", 0))
        )

    assert [node.data for node in excinfo.value.cycle] == ["KSampler_4", "KSampler_4"]


@pytest.mark.parametrize(
    ("latent_image", "error"),
    [(("LatentNode_9", 0), NodeNotFoundError), (("KSampler_4", 0), CycleError)],
    ids=["missing-node", "own-output"],
)
def test_rejected_node_leaves_the_workflow_unchanged(latent_image, error):
    workflow = Workflow()
    sources = add_sources(workflow)
    before = state(workflow)

    with pytest.raises(error):
        workflow.add_node(
            KSampler, **sampler_inputs(sources, latent_image=latent_image)
        )

    assert state(workflow) == before
    with pytest.raises(NodeNotFoundError):
        workflow.dependencies("KSampler_4")


def test_rewiring_an_input_moves_the_dependency():
    workflow = Workflow()
    sources = add_sources(workflow)
    sampler = workflow.add_node(KSampler, **sampler_inputs(sources))
    other_latent = workflow.add_node(LatentNode)

    sampler.latent_image = other_latent.output["LATENT"]

    assert other_latent.node_id in workflow.dependencies(sampler.node_id)
    assert workflow.dependents(sources.latent.node_id) == set()


def test_dependency_stays_while_another_input_still_uses_the_node():
    workflow = Workflow()
    sources = add_sources(workflow)
    switch = workflow.add_node(
        ComfySwitchNode,
        switch=True,
        on_true=sources.positive.output["CONDITIONING"],
        on_false=sources.positive.output["CONDITIONING"],
    )

    switch.on_false = sources.negative.output["CONDITIONING"]

    assert sources.positive.node_id in workflow.dependencies(switch.node_id)


def test_rewiring_that_closes_a_cycle_is_rejected():
    workflow = Workflow()
    sources = add_sources(workflow)
    first = workflow.add_node(KSampler, **sampler_inputs(sources))
    second = workflow.add_node(
        KSampler, **sampler_inputs(sources, latent_image=first.output["LATENT"])
    )

    with pytest.raises(InvalidNodeChangeError) as excinfo:
        first.latent_image = second.output["LATENT"]

    assert str(excinfo.value) == (
        "Cannot change 'latent_image' of node 'KSampler_4': node 'KSampler_5' "
        "depends on 'KSampler_4', so the change would create the cycle "
        "'KSampler_4' -> 'KSampler_5' -> 'KSampler_4'."
    )
    assert [node.data for node in excinfo.value.__cause__.cycle] == [
        first.node_id,
        second.node_id,
        first.node_id,
    ]


def test_node_using_its_own_output_after_creation_is_rejected():
    workflow = Workflow()
    sampler = workflow.add_node(KSampler, **sampler_inputs(add_sources(workflow)))

    with pytest.raises(InvalidNodeChangeError) as excinfo:
        sampler.latent_image = sampler.output["LATENT"]

    assert str(excinfo.value) == (
        "Cannot change 'latent_image' of node 'KSampler_4': "
        "a node cannot use its own output."
    )


def test_rewiring_to_a_node_outside_the_workflow_is_rejected():
    workflow = Workflow()
    sampler = workflow.add_node(KSampler, **sampler_inputs(add_sources(workflow)))

    with pytest.raises(InvalidNodeChangeError) as excinfo:
        sampler.latent_image = ("LatentNode_9", 0)

    assert str(excinfo.value) == (
        "Cannot change 'latent_image' of node 'KSampler_4': "
        "node 'LatentNode_9' is not in the workflow."
    )


@pytest.mark.parametrize(
    "make_link",
    [lambda second: second.output["LATENT"], lambda second: ("LatentNode_9", 0)],
    ids=["cycle", "missing-node"],
)
def test_rejected_rewiring_keeps_the_old_input_and_the_dependencies(make_link):
    workflow = Workflow()
    sources = add_sources(workflow)
    first = workflow.add_node(KSampler, **sampler_inputs(sources))
    second = workflow.add_node(
        KSampler, **sampler_inputs(sources, latent_image=first.output["LATENT"])
    )
    before = state(workflow)

    with pytest.raises(InvalidNodeChangeError):
        first.latent_image = make_link(second)

    assert first.latent_image == sources.latent.output["LATENT"]
    assert state(workflow) == before


def test_rejected_connection_on_an_unset_input_leaves_it_unset():
    workflow = Workflow()
    node = workflow.add_node(MaskNode)

    with pytest.raises(InvalidNodeChangeError):
        node.mask = ("LatentNode_9", 0)

    assert not hasattr(node, "mask")


def test_deleting_a_connection_input_drops_the_dependency():
    workflow = Workflow()
    sources = add_sources(workflow)
    node = workflow.add_node(MaskNode)
    node.mask = sources.latent.output["LATENT"]

    del node.mask

    assert workflow.dependencies(node.node_id) == set()


def test_input_that_is_not_optional_cannot_be_deleted_from_a_workflow_node():
    workflow = Workflow()
    sources = add_sources(workflow)
    sampler = workflow.add_node(KSampler, **sampler_inputs(sources))
    before = state(workflow)

    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        del sampler.latent_image

    assert str(excinfo.value) == (
        "latent_image cannot be removed because it is not optional"
    )
    assert state(workflow) == before


def test_changing_a_value_input_keeps_the_dependencies():
    workflow = Workflow()
    sources = add_sources(workflow)
    sampler = workflow.add_node(KSampler, **sampler_inputs(sources))
    before = workflow.dependencies(sampler.node_id)

    sampler.steps = 30

    assert workflow.dependencies(sampler.node_id) == before


@pytest.mark.parametrize("query", ["dependencies", "dependents"])
def test_dependency_queries_for_unknown_node_raise_node_not_found(query):
    workflow = Workflow()

    with pytest.raises(NodeNotFoundError, match="Missing_0"):
        getattr(workflow, query)("Missing_0")


def test_removed_node_leaves_the_workflow():
    workflow = Workflow()
    sources = add_sources(workflow)
    sampler = workflow.add_node(KSampler, **sampler_inputs(sources))

    workflow.remove_node(sampler.node_id)

    assert sampler.node_id not in workflow.nodes
    assert sampler.node_id not in workflow.get_comfy_ui_workflow_format()


def test_removed_node_is_no_longer_a_dependent_of_the_nodes_it_used():
    workflow = Workflow()
    sources = add_sources(workflow)
    sampler = workflow.add_node(KSampler, **sampler_inputs(sources))

    workflow.remove_node(sampler.node_id)

    assert {node.node_id: workflow.dependents(node.node_id) for node in sources} == {
        sources.checkpoint.node_id: {
            sources.positive.node_id,
            sources.negative.node_id,
        },
        sources.positive.node_id: set(),
        sources.negative.node_id: set(),
        sources.latent.node_id: set(),
    }


@pytest.mark.parametrize(
    "query", ["get_node_output", "dependencies", "dependents", "remove_node"]
)
def test_removed_node_id_is_unknown(query):
    workflow = Workflow()
    node = workflow.add_node(LatentNode)
    workflow.remove_node(node.node_id)

    with pytest.raises(NodeNotFoundError, match=node.node_id):
        getattr(workflow, query)(node.node_id)


def test_connection_to_a_removed_node_is_rejected():
    workflow = Workflow()
    sources = add_sources(workflow)
    workflow.remove_node(sources.latent.node_id)

    with pytest.raises(NodeNotFoundError, match=sources.latent.node_id):
        workflow.add_node(KSampler, **sampler_inputs(sources))


def test_changes_to_a_removed_node_do_not_reach_the_workflow():
    workflow = Workflow()
    sources = add_sources(workflow)
    sampler = workflow.add_node(KSampler, **sampler_inputs(sources))
    workflow.remove_node(sampler.node_id)
    before = state(workflow)

    sampler.latent_image = sources.latent.output["LATENT"]

    assert state(workflow) == before


def test_ids_of_removed_nodes_are_not_reused():
    workflow = Workflow()
    removed = workflow.add_node(LatentNode)
    workflow.remove_node(removed.node_id)

    added = workflow.add_node(LatentNode)

    assert added.node_id != removed.node_id


def test_removing_an_unknown_node_raises_node_not_found():
    workflow = Workflow()

    with pytest.raises(NodeNotFoundError, match="Missing_0"):
        workflow.remove_node("Missing_0")


def test_node_with_dependents_cannot_be_removed():
    workflow = Workflow()
    sources = add_sources(workflow)
    workflow.add_node(KSampler, **sampler_inputs(sources))

    with pytest.raises(NodeHasDependentsError) as excinfo:
        workflow.remove_node(sources.latent.node_id)

    assert (
        str(excinfo.value)
        == "Cannot remove node 'LatentNode_3': node 'KSampler_4' depends on it."
    )


def test_node_has_dependents_error_lists_every_dependent_in_creation_order():
    workflow = Workflow()
    sources = add_sources(workflow)
    workflow.add_node(KSampler, **sampler_inputs(sources))

    with pytest.raises(NodeHasDependentsError) as excinfo:
        workflow.remove_node(sources.checkpoint.node_id)

    assert excinfo.value.node_id == "CheckpointLoaderSimple_0"
    assert excinfo.value.dependents == ["TextNode_1", "TextNode_2", "KSampler_4"]
    assert str(excinfo.value) == (
        "Cannot remove node 'CheckpointLoaderSimple_0': "
        "nodes 'TextNode_1', 'TextNode_2' and 'KSampler_4' depend on it."
    )


def test_rejected_removal_leaves_the_workflow_unchanged():
    workflow = Workflow()
    sources = add_sources(workflow)
    workflow.add_node(KSampler, **sampler_inputs(sources))
    before = state(workflow)

    with pytest.raises(NodeHasDependentsError):
        workflow.remove_node(sources.checkpoint.node_id)

    assert state(workflow) == before


def test_node_kept_after_a_rejected_removal_is_still_checked():
    workflow = Workflow()
    sources = add_sources(workflow)
    first = workflow.add_node(KSampler, **sampler_inputs(sources))
    second = workflow.add_node(
        KSampler, **sampler_inputs(sources, latent_image=first.output["LATENT"])
    )

    with pytest.raises(NodeHasDependentsError):
        workflow.remove_node(first.node_id)

    with pytest.raises(InvalidNodeChangeError):
        first.latent_image = second.output["LATENT"]


def test_node_has_dependents_error_is_node_exception():
    assert issubclass(NodeHasDependentsError, NodeException)


@pytest.mark.parametrize(
    "clone",
    [lambda error: pickle.loads(pickle.dumps(error)), copy.copy],
    ids=["pickle", "copy"],
)
def test_node_has_dependents_error_survives_pickle_and_copy(clone):
    error = NodeHasDependentsError("LatentNode_3", ["KSampler_4"])

    restored = clone(error)

    assert str(restored) == str(error)
    assert (restored.node_id, restored.dependents) == ("LatentNode_3", ["KSampler_4"])


def test_id_of_a_node_with_dependents_cannot_change():
    workflow = Workflow()
    sources = add_sources(workflow)
    workflow.add_node(KSampler, **sampler_inputs(sources))

    with pytest.raises(InvalidNodeChangeError) as excinfo:
        sources.checkpoint.node_id = "Loader"

    assert sources.checkpoint.node_id == "CheckpointLoaderSimple_0"
    assert str(excinfo.value) == (
        "Cannot change 'node_id' of node 'CheckpointLoaderSimple_0': "
        "nodes 'TextNode_1', 'TextNode_2' and 'KSampler_4' depend on it."
    )


def test_id_of_a_node_in_a_workflow_cannot_change():
    workflow = Workflow()
    node = workflow.add_node(LatentNode)
    before = state(workflow)

    with pytest.raises(InvalidNodeChangeError) as excinfo:
        node.node_id = "Latent"

    assert state(workflow) == before
    assert str(excinfo.value) == (
        "Cannot change 'node_id' of node 'LatentNode_0': "
        "a node keeps its id while it is in a workflow."
    )


def test_update_node_changes_the_given_inputs():
    workflow = Workflow()
    sources = add_sources(workflow)
    sampler = workflow.add_node(KSampler, **sampler_inputs(sources))
    other_latent = workflow.add_node(LatentNode)

    workflow.update_node(
        sampler.node_id, steps=30, latent_image=other_latent.output["LATENT"]
    )

    inputs = workflow.get_comfy_ui_workflow_format()[sampler.node_id]["inputs"]
    assert (inputs["steps"], inputs["latent_image"]) == (30, ("LatentNode_5", 0))
    assert workflow.dependents(other_latent.node_id) == {sampler.node_id}


def test_changes_after_update_node_are_still_validated():
    workflow = Workflow()
    sources = add_sources(workflow)
    sampler = workflow.add_node(KSampler, **sampler_inputs(sources))
    workflow.update_node(sampler.node_id, steps=30)

    with pytest.raises(InvalidNodeParameterValue, match="steps"):
        sampler.steps = 0

    assert sampler.steps == 30


def test_add_node_rejects_the_same_link_for_positive_and_negative():
    workflow = Workflow()
    sources = add_sources(workflow)
    before = state(workflow)

    with pytest.raises(InvalidNodeParameterValue):
        workflow.add_node(
            KSampler,
            **sampler_inputs(sources, negative=sources.positive.output["CONDITIONING"]),
        )

    assert state(workflow) == before


def test_update_node_rejects_the_same_link_for_positive_and_negative():
    workflow = Workflow()
    sources = add_sources(workflow)
    sampler = workflow.add_node(KSampler, **sampler_inputs(sources))
    before = state(workflow)

    with pytest.raises(InvalidNodeParameterValue):
        workflow.update_node(
            sampler.node_id, negative=sources.positive.output["CONDITIONING"]
        )

    assert state(workflow) == before


def test_update_node_can_swap_positive_and_negative():
    workflow = Workflow()
    sources = add_sources(workflow)
    sampler = workflow.add_node(KSampler, **sampler_inputs(sources))
    positive = sources.positive.output["CONDITIONING"]
    negative = sources.negative.output["CONDITIONING"]

    workflow.update_node(sampler.node_id, positive=negative, negative=positive)

    assert (sampler.positive, sampler.negative) == (negative, positive)


def test_update_node_can_connect_an_unset_input():
    workflow = Workflow()
    latent = workflow.add_node(LatentNode)
    node = workflow.add_node(MaskNode)

    workflow.update_node(node.node_id, mask=latent.output["LATENT"])

    assert workflow.dependencies(node.node_id) == {latent.node_id}


@pytest.mark.parametrize(
    "make_link",
    [lambda second: second.output["LATENT"], lambda second: ("LatentNode_9", 0)],
    ids=["cycle", "missing-node"],
)
def test_rejected_update_changes_none_of_the_inputs(make_link):
    workflow = Workflow()
    sources = add_sources(workflow)
    first = workflow.add_node(KSampler, **sampler_inputs(sources))
    second = workflow.add_node(
        KSampler, **sampler_inputs(sources, latent_image=first.output["LATENT"])
    )
    before = state(workflow)

    with pytest.raises(InvalidNodeChangeError, match="'latent_image'"):
        workflow.update_node(
            first.node_id,
            steps=30,
            positive=sources.negative.output["CONDITIONING"],
            latent_image=make_link(second),
        )

    assert state(workflow) == before


def test_rejected_update_leaves_unset_inputs_unset():
    workflow = Workflow()
    latent = workflow.add_node(LatentNode)
    node = workflow.add_node(MaskNode)

    with pytest.raises(InvalidNodeChangeError, match="'image'"):
        workflow.update_node(
            node.node_id, mask=latent.output["LATENT"], image=("LatentNode_9", 0)
        )

    assert not hasattr(node, "mask")
    assert workflow.dependents(latent.node_id) == set()


@pytest.mark.parametrize("name", ["stpes", "output", "_on_change"])
def test_update_node_rejects_names_that_are_not_inputs(name):
    workflow = Workflow()
    sampler = workflow.add_node(KSampler, **sampler_inputs(add_sources(workflow)))
    before = state(workflow)

    with pytest.raises(InvalidNodeChangeError) as excinfo:
        workflow.update_node(sampler.node_id, steps=30, **{name: None})

    assert str(excinfo.value) == (
        f"Cannot change {name!r} of node 'KSampler_4': KSampler has no input {name!r}."
    )
    assert state(workflow) == before


def test_node_is_still_checked_after_a_rejected_update():
    workflow = Workflow()
    sampler = workflow.add_node(KSampler, **sampler_inputs(add_sources(workflow)))

    with pytest.raises(InvalidNodeChangeError):
        workflow.update_node(sampler.node_id, _on_change=None)

    with pytest.raises(InvalidNodeChangeError):
        sampler.latent_image = sampler.output["LATENT"]


def test_update_node_with_a_value_of_another_type_changes_nothing():
    workflow = Workflow()
    sources = add_sources(workflow)
    sampler = workflow.add_node(KSampler, **sampler_inputs(sources))
    before = state(workflow)

    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        workflow.update_node(sampler.node_id, steps=30, cfg="high")

    assert str(excinfo.value) == "cfg must be a number or a link, not str"
    assert state(workflow) == before


def test_update_node_for_unknown_node_raises_node_not_found():
    workflow = Workflow()

    with pytest.raises(NodeNotFoundError, match="Missing_0"):
        workflow.update_node("Missing_0", steps=30)


def test_invalid_node_change_error_is_node_exception():
    assert issubclass(InvalidNodeChangeError, NodeException)


@pytest.mark.parametrize(
    "clone",
    [lambda error: pickle.loads(pickle.dumps(error)), copy.copy],
    ids=["pickle", "copy"],
)
def test_invalid_node_change_error_survives_pickle_and_copy(clone):
    error = InvalidNodeChangeError("KSampler_4", "latent_image", "a reason")

    restored = clone(error)

    assert str(restored) == str(error)
    assert (restored.node_id, restored.name, restored.reason) == (
        "KSampler_4",
        "latent_image",
        "a reason",
    )


def typed_sources(workflow):
    checkpoint = workflow.add_node(
        CheckpointLoaderSimple, ckpt_name="model.safetensors"
    )
    text = workflow.add_node(
        CLIPTextEncode, text="a cat", clip=checkpoint.output["CLIP"]
    )
    latent = workflow.add_node(EmptyLatentImage)
    negative = workflow.add_node(
        CLIPTextEncode, text="blurry", clip=checkpoint.output["CLIP"]
    )
    return checkpoint, text, latent, negative


def typed_sampler_inputs(checkpoint, text, latent, negative, **overrides):
    return {
        "model": checkpoint.output["MODEL"],
        "sampler_name": "euler",
        "scheduler": "normal",
        "positive": text.output["CONDITIONING"],
        "negative": negative.output["CONDITIONING"],
        "latent_image": latent.output["LATENT"],
        **overrides,
    }


def test_add_node_rejects_a_link_whose_output_type_the_input_does_not_take():
    workflow = Workflow()
    checkpoint, text, _, _ = typed_sources(workflow)
    before = state(workflow)

    with pytest.raises(InvalidLinkError) as excinfo:
        workflow.add_node(
            VAEDecode,
            samples=text.output["CONDITIONING"],
            vae=checkpoint.output["VAE"],
        )

    assert str(excinfo.value) == (
        "Cannot add node 'VAEDecode_4': input 'samples' uses output 0 of node "
        "'CLIPTextEncode_1', which is CONDITIONING, but the input takes LATENT."
    )
    assert state(workflow) == before
    with pytest.raises(NodeNotFoundError):
        workflow.dependencies("VAEDecode_3")


def test_add_node_rejects_a_link_to_an_output_the_source_does_not_have():
    workflow = Workflow()
    checkpoint, _, _, _ = typed_sources(workflow)

    with pytest.raises(InvalidLinkError) as excinfo:
        workflow.add_node(CLIPTextEncode, text="a dog", clip=(checkpoint.node_id, 5))

    assert str(excinfo.value) == (
        "Cannot add node 'CLIPTextEncode_4': input 'clip' uses output 5 of node "
        "'CheckpointLoaderSimple_0', but CheckpointLoaderSimple has no output 5."
    )


def test_widget_input_takes_a_link_of_its_own_type_only():
    workflow = Workflow()
    sources = typed_sources(workflow)
    steps = workflow.add_node(PrimitiveInt, value=30)
    sampler = workflow.add_node(
        KSampler, **typed_sampler_inputs(*sources, steps=steps.output["INT"])
    )

    with pytest.raises(InvalidLinkError) as excinfo:
        workflow.add_node(
            KSampler, **typed_sampler_inputs(*sources, cfg=steps.output["INT"])
        )

    assert steps.node_id in workflow.dependencies(sampler.node_id)
    assert str(excinfo.value) == (
        "Cannot add node 'KSampler_6': input 'cfg' uses output 0 of node "
        "'PrimitiveInt_4', which is INT, but the input takes FLOAT."
    )


def test_any_and_match_type_sockets_take_every_output():
    workflow = Workflow()
    checkpoint, text, latent, negative = typed_sources(workflow)

    switch = workflow.add_node(
        ComfySwitchNode,
        switch=True,
        on_true=checkpoint.output["MODEL"],
        on_false=checkpoint.output["MODEL"],
    )
    sampler = workflow.add_node(
        KSampler,
        **typed_sampler_inputs(
            checkpoint, text, latent, negative, model=switch.output["output"]
        ),
    )
    preview = workflow.add_node(PreviewAny, source=sampler.output["LATENT"])

    assert workflow.dependencies(preview.node_id) == {sampler.node_id}
    assert switch.node_id in workflow.dependencies(sampler.node_id)


def test_union_socket_takes_each_of_its_types():
    workflow = Workflow()
    checkpoint, text, _, _ = typed_sources(workflow)
    vae = checkpoint.output["VAE"]
    integer = workflow.add_node(PrimitiveInt, value=24)
    real = workflow.add_node(PrimitiveFloat, value=23.976)

    workflow.add_node(
        LTXVEmptyLatentAudio, audio_vae=vae, frame_rate=integer.output["INT"]
    )
    workflow.add_node(
        LTXVEmptyLatentAudio, audio_vae=vae, frame_rate=real.output["FLOAT"]
    )
    with pytest.raises(InvalidLinkError) as excinfo:
        workflow.add_node(
            LTXVEmptyLatentAudio,
            audio_vae=vae,
            frame_rate=text.output["CONDITIONING"],
        )

    assert str(excinfo.value).endswith(
        "which is CONDITIONING, but the input takes FLOAT,INT."
    )


def video_parts(workflow):
    image = workflow.add_node(EmptyImage, width=8, height=8)
    video = workflow.add_node(CreateVideo, images=image.output["IMAGE"])
    return workflow.add_node(GetVideoComponents, video=video.output["VIDEO"])


def test_v1_combo_does_not_take_a_combo_output_as_comfy_ui():
    workflow = Workflow()
    sources = typed_sources(workflow)
    parts = video_parts(workflow)
    before = state(workflow)

    with pytest.raises(InvalidLinkError) as excinfo:
        workflow.add_node(
            KSampler,
            **typed_sampler_inputs(*sources, sampler_name=parts.output["bit_depth"]),
        )

    assert str(excinfo.value) == (
        "Cannot add node 'KSampler_7': input 'sampler_name' uses output 3 of node "
        "'GetVideoComponents_6', which is COMBO, but the input takes V1 COMBO."
    )
    assert state(workflow) == before


def test_v1_combo_takes_any_and_match_type_outputs():
    workflow = Workflow()
    sources = typed_sources(workflow)
    name = workflow.add_node(PrimitiveString, value="euler")
    item = workflow.add_node(GetItemFromList, list=name.output["STRING"])
    switch = workflow.add_node(
        ComfySwitchNode,
        switch=True,
        on_true=name.output["STRING"],
        on_false=name.output["STRING"],
    )

    by_item = workflow.add_node(
        KSampler, **typed_sampler_inputs(*sources, sampler_name=item.output["*"])
    )
    by_switch = workflow.add_node(
        KSampler, **typed_sampler_inputs(*sources, scheduler=switch.output["output"])
    )

    assert item.node_id in workflow.dependencies(by_item.node_id)
    assert switch.node_id in workflow.dependencies(by_switch.node_id)


def test_v3_combo_takes_a_combo_output():
    workflow = Workflow()
    parts = video_parts(workflow)

    video = workflow.add_node(
        CreateVideo,
        images=parts.output["images"],
        bit_depth=parts.output["bit_depth"],
        color_space=parts.output["color_space"],
    )

    assert workflow.dependencies(video.node_id) == {parts.node_id}


def test_update_node_rejects_a_combo_output_into_a_v1_combo():
    workflow = Workflow()
    sources = typed_sources(workflow)
    sampler = workflow.add_node(KSampler, **typed_sampler_inputs(*sources))
    parts = video_parts(workflow)
    before = state(workflow)

    with pytest.raises(InvalidNodeChangeError) as excinfo:
        workflow.update_node(sampler.node_id, scheduler=parts.output["color_space"])

    assert "which is COMBO, but the input takes V1 COMBO" in str(excinfo.value)
    assert state(workflow) == before


class SpacedSocketNode(BaseNode):
    __slots__ = ("value",)
    CATEGORY = "test"
    INPUT_SOCKET_TYPES = {"value": "FLOAT, INT"}

    def __init__(self, node_id, value):
        super().__init__(node_id=node_id)
        self.value = value


def test_socket_types_are_read_without_the_spaces_around_commas():
    workflow = Workflow()
    integer = workflow.add_node(PrimitiveInt, value=24)

    node = workflow.add_node(SpacedSocketNode, value=integer.output["INT"])

    assert workflow.dependencies(node.node_id) == {integer.node_id}


def test_nodes_without_socket_types_are_not_type_checked():
    workflow = Workflow()
    sources = add_sources(workflow)

    node = workflow.add_node(
        TextNode, text="a dog", clip=sources.checkpoint.output["MODEL"]
    )

    assert workflow.dependencies(node.node_id) == {sources.checkpoint.node_id}


def test_update_node_rejects_a_link_whose_output_type_the_input_does_not_take():
    workflow = Workflow()
    checkpoint, text, _, _ = typed_sources(workflow)
    before = state(workflow)

    with pytest.raises(InvalidNodeChangeError) as excinfo:
        workflow.update_node(
            text.node_id, text="a dog", clip=checkpoint.output["MODEL"]
        )

    assert str(excinfo.value) == (
        "Cannot change 'clip' of node 'CLIPTextEncode_1': it would use output 0 of "
        "node 'CheckpointLoaderSimple_0', which is MODEL, but the input takes CLIP."
    )
    assert state(workflow) == before


def test_assigning_a_link_to_an_output_the_source_does_not_have_is_rejected():
    workflow = Workflow()
    checkpoint, text, _, _ = typed_sources(workflow)

    with pytest.raises(InvalidNodeChangeError) as excinfo:
        text.clip = (checkpoint.node_id, 3)

    assert str(excinfo.value) == (
        "Cannot change 'clip' of node 'CLIPTextEncode_1': it would use output 3 of "
        "node 'CheckpointLoaderSimple_0', but CheckpointLoaderSimple has no output 3."
    )
    assert text.clip == checkpoint.output["CLIP"]


def test_invalid_link_error_is_node_exception():
    assert issubclass(InvalidLinkError, NodeException)


@pytest.mark.parametrize(
    "clone",
    [lambda error: pickle.loads(pickle.dumps(error)), copy.copy],
    ids=["pickle", "copy"],
)
def test_invalid_link_error_survives_pickle_and_copy(clone):
    error = InvalidLinkError("KSampler_4", "model", "a reason")

    restored = clone(error)

    assert str(restored) == str(error)
    assert (restored.node_id, restored.name, restored.reason) == (
        "KSampler_4",
        "model",
        "a reason",
    )


class RangeNode(BaseNode):
    __slots__ = ("low", "high")
    CATEGORY = "test"

    def __init__(self, node_id: str, low: int = 0, high: int = 1) -> None:
        super().__init__(node_id=node_id)
        self.low = low
        self.high = high
        if self.low > self.high:
            raise InvalidNodeParameterValue("low must not be above high")


def test_update_node_validates_the_node_once_with_every_change():
    workflow = Workflow()
    node = workflow.add_node(RangeNode)

    workflow.update_node(node.node_id, low=5, high=10)

    assert (node.low, node.high) == (5, 10)


def test_update_node_whose_result_breaks_the_node_validation_changes_nothing():
    workflow = Workflow()
    node = workflow.add_node(RangeNode)
    before = state(workflow)

    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        workflow.update_node(node.node_id, high=10, low=20)

    assert str(excinfo.value) == "low must not be above high"
    assert state(workflow) == before


def test_update_node_with_a_value_out_of_range_changes_nothing():
    workflow = Workflow()
    sources = add_sources(workflow)
    sampler = workflow.add_node(KSampler, **sampler_inputs(sources))
    before = state(workflow)

    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        workflow.update_node(sampler.node_id, cfg=7.0, steps=0)

    assert str(excinfo.value) == "steps must be in closed range: [1, 10000]"
    assert state(workflow) == before
