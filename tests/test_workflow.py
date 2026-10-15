import pytest

from comfy_ui_api_workflow._impl.enums import OutputTypes
from comfy_ui_api_workflow._impl.errors.node import NodeNotHaveOutput
from comfy_ui_api_workflow._impl.exceptions import NodeException
from comfy_ui_api_workflow._impl.nodes.base import BaseNode
from comfy_ui_api_workflow._impl.nodes.model.sampling import KSampler
from comfy_ui_api_workflow._impl.workflow import Workflow


class OutputlessNode(BaseNode):
    def get_comfy_ui_workflow_format(self):
        return {"class_type": "OutputlessNode", "inputs": {}}


def test_new_workflow_is_empty():
    workflow = Workflow()

    assert workflow.comfy_ui_workflow == {}
    assert workflow.nodes == {}


def test_add_node_returns_class_name_with_index(ksampler_inputs):
    workflow = Workflow()

    first = workflow.add_node(KSampler, **ksampler_inputs)
    second = workflow.add_node(KSampler, **ksampler_inputs)

    assert (first, second) == ("KSampler_0", "KSampler_1")


def test_add_node_registers_node_and_comfy_ui_format(ksampler_inputs):
    workflow = Workflow()

    node_id = workflow.add_node(KSampler, **ksampler_inputs)

    node = workflow.nodes[node_id]
    assert isinstance(node, KSampler)
    assert node.node_id == node_id
    assert workflow.comfy_ui_workflow[node_id] == node.get_comfy_ui_workflow_format()


def test_add_ksampler_returns_class_name_with_index(ksampler_inputs):
    workflow = Workflow()

    assert workflow.add_ksampler(**ksampler_inputs) == "KSampler_0"


def test_add_ksampler_stores_comfy_ui_format(ksampler_inputs):
    workflow = Workflow()

    node_id = workflow.add_ksampler(**ksampler_inputs, seed=42, steps=30)

    entry = workflow.comfy_ui_workflow[node_id]
    assert entry == workflow.nodes[node_id].get_comfy_ui_workflow_format()
    assert entry["inputs"]["seed"] == 42
    assert entry["inputs"]["steps"] == 30


def test_indices_are_shared_across_add_methods(ksampler_inputs):
    workflow = Workflow()

    first = workflow.add_node(KSampler, **ksampler_inputs)
    second = workflow.add_ksampler(**ksampler_inputs)

    assert (first, second) == ("KSampler_0", "KSampler_1")


def test_nodes_can_be_chained_through_outputs(ksampler_inputs):
    workflow = Workflow()
    first = workflow.add_ksampler(**ksampler_inputs)

    second = workflow.add_ksampler(
        **{**ksampler_inputs, "latent_image": workflow.get_node_output(first)}
    )

    assert workflow.comfy_ui_workflow[second]["inputs"]["latent_image"] == [first, 0]


def test_get_node_output_returns_node_output(ksampler_inputs):
    workflow = Workflow()
    node_id = workflow.add_ksampler(**ksampler_inputs)

    output = workflow.get_node_output(node_id)

    assert output.get(OutputTypes.LATENT) == [node_id, 0]


def test_get_node_output_for_node_without_output_raises():
    workflow = Workflow()
    node_id = workflow.add_node(OutputlessNode)

    with pytest.raises(NodeNotHaveOutput, match=node_id):
        workflow.get_node_output(node_id)


def test_get_node_output_for_unknown_node_raises_library_exception():
    workflow = Workflow()

    with pytest.raises(NodeException):
        workflow.get_node_output("Missing0")
