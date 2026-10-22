import copy
import importlib
import json
import pickle
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import comfy_ui_api_workflow._impl.nodes as nodes_package
from comfy_ui_api_workflow._impl.errors.data_structure import CycleError
from comfy_ui_api_workflow._impl.errors.node import (
    InvalidNodeChangeError,
    InvalidNodeParameterValue,
    NodeHasDependentsError,
    NodeNotFoundError,
)
from comfy_ui_api_workflow._impl.errors.workflow import InvalidWorkflowError
from comfy_ui_api_workflow._impl.exceptions import WorkflowException
from comfy_ui_api_workflow._impl.nodes.base import BaseNode
from comfy_ui_api_workflow._impl.nodes.image import SaveImage
from comfy_ui_api_workflow._impl.nodes.model.conditioning import CLIPTextEncode
from comfy_ui_api_workflow._impl.nodes.model.latent import EmptyLatentImage, VAEDecode
from comfy_ui_api_workflow._impl.nodes.model.loaders import CheckpointLoaderSimple
from comfy_ui_api_workflow._impl.nodes.model.patch.unet import EpsilonScaling
from comfy_ui_api_workflow._impl.nodes.model.sampling import KSampler
from comfy_ui_api_workflow._impl.workflow import Workflow

WAN_WORKFLOW = Path(__file__).resolve().parent.parent / "video_wan2_2_14B_i2v.json"

# The default workflow, as sent by ComfyUI's script_examples/basic_api_example.py.
BASIC_API_EXAMPLE = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "cfg": 8,
            "denoise": 1,
            "latent_image": ["5", 0],
            "model": ["4", 0],
            "negative": ["7", 0],
            "positive": ["6", 0],
            "sampler_name": "euler",
            "scheduler": "normal",
            "seed": 8566257,
            "steps": 20,
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"},
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"batch_size": 1, "height": 512, "width": 512},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["4", 1], "text": "masterpiece best quality girl"},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["4", 1], "text": "bad hands"},
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
    },
}

MOGE_RENDER = {
    "1": {"class_type": "LoadMoGeModel", "inputs": {"model_name": "moge.safetensors"}},
    "2": {"class_type": "LoadImage", "inputs": {"image": "photo.png"}},
    "3": {
        "class_type": "MoGeInference",
        "inputs": {"moge_model": ["1", 0], "image": ["2", 0]},
    },
    "4": {
        "class_type": "MoGeRender",
        "inputs": {"moge_geometry": ["3", 0], "output": "normal_opengl"},
    },
}


def with_inputs(node_id, **inputs):
    workflow_format = copy.deepcopy(BASIC_API_EXAMPLE)
    workflow_format[node_id]["inputs"].update(inputs)
    return workflow_format


def without_inputs(node_id, *names):
    workflow_format = copy.deepcopy(BASIC_API_EXAMPLE)
    for name in names:
        del workflow_format[node_id]["inputs"][name]
    return workflow_format


IMG2IMG_LOOP = {
    **with_inputs("3", latent_image=["10", 0]),
    "10": {"class_type": "VAEEncode", "inputs": {"pixels": ["8", 0], "vae": ["4", 2]}},
}


def loaded(workflow_format):
    workflow = Workflow()
    workflow.load_workflow(workflow_format)
    return workflow


def exported(workflow):
    return json.loads(json.dumps(workflow.get_comfy_ui_workflow_format()))


def linked_sources(workflow_format):
    return {
        node_id: {
            value[0] for value in node["inputs"].values() if isinstance(value, list)
        }
        for node_id, node in workflow_format.items()
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


def test_load_workflow_rebuilds_the_comfy_ui_basic_api_example():
    workflow = loaded(BASIC_API_EXAMPLE)

    assert exported(workflow) == BASIC_API_EXAMPLE


def test_loaded_nodes_keep_the_ids_and_class_types_of_the_workflow():
    workflow = loaded(BASIC_API_EXAMPLE)

    assert {node_id: type(node) for node_id, node in workflow.nodes.items()} == {
        "3": KSampler,
        "4": CheckpointLoaderSimple,
        "5": EmptyLatentImage,
        "6": CLIPTextEncode,
        "7": CLIPTextEncode,
        "8": VAEDecode,
        "9": SaveImage,
    }
    assert list(workflow.nodes) == list(BASIC_API_EXAMPLE)
    assert all(node.node_id == node_id for node_id, node in workflow.nodes.items())


def test_loaded_links_are_outputs_of_the_loaded_nodes():
    workflow = loaded(BASIC_API_EXAMPLE)
    checkpoint = workflow.nodes["4"]

    assert workflow.nodes["3"].model == checkpoint.output["MODEL"]
    assert workflow.nodes["6"].clip == checkpoint.output["CLIP"]
    assert workflow.nodes["8"].vae == checkpoint.output["VAE"]


def test_loaded_links_are_dependencies_even_on_nodes_listed_after():
    workflow = loaded(BASIC_API_EXAMPLE)

    assert {
        node_id: workflow.dependencies(node_id) for node_id in workflow.nodes
    } == linked_sources(BASIC_API_EXAMPLE)
    assert workflow.dependents("4") == {"3", "6", "7", "8"}
    assert workflow.dependents("9") == set()


def test_workflow_built_with_add_node_is_loaded_back():
    built = Workflow()
    checkpoint = built.add_node(CheckpointLoaderSimple, ckpt_name="model.safetensors")
    positive = built.add_node(
        CLIPTextEncode, text="a cat", clip=checkpoint.output["CLIP"]
    )
    negative = built.add_node(
        CLIPTextEncode, text="blurry", clip=checkpoint.output["CLIP"]
    )
    latent = built.add_node(EmptyLatentImage, width=512, height=512, batch_size=1)
    built.add_node(
        KSampler,
        model=checkpoint.output["MODEL"],
        sampler_name="euler",
        scheduler="normal",
        positive=positive.output["CONDITIONING"],
        negative=negative.output["CONDITIONING"],
        latent_image=latent.output["LATENT"],
    )

    workflow = loaded(built.get_comfy_ui_workflow_format())

    assert exported(workflow) == exported(built)
    assert state(workflow)[2] == state(built)[2]


def test_loaded_node_without_links_is_in_the_workflow_graph():
    workflow = loaded({"4": BASIC_API_EXAMPLE["4"]})

    assert (workflow.dependencies("4"), workflow.dependents("4")) == (set(), set())
    workflow.remove_node("4")
    assert workflow.nodes == {}


def test_load_workflow_replaces_the_current_workflow():
    workflow = Workflow()
    workflow.add_node(CheckpointLoaderSimple, ckpt_name="old.safetensors")

    workflow.load_workflow(BASIC_API_EXAMPLE)

    assert list(workflow.nodes) == list(BASIC_API_EXAMPLE)
    assert exported(workflow) == BASIC_API_EXAMPLE
    with pytest.raises(NodeNotFoundError):
        workflow.dependencies("CheckpointLoaderSimple_0")


def test_loading_an_empty_workflow_leaves_the_instance_empty():
    workflow = loaded(BASIC_API_EXAMPLE)

    workflow.load_workflow({})

    assert workflow.nodes == {}
    assert workflow.get_comfy_ui_workflow_format() == {}
    node = workflow.add_node(CheckpointLoaderSimple, ckpt_name="model.safetensors")
    assert node.node_id == "CheckpointLoaderSimple_0"


def test_nodes_of_the_replaced_workflow_no_longer_affect_it():
    workflow = Workflow()
    checkpoint = workflow.add_node(CheckpointLoaderSimple, ckpt_name="old.safetensors")
    text = workflow.add_node(CLIPTextEncode, text="old", clip=checkpoint.output["CLIP"])
    workflow.load_workflow(BASIC_API_EXAMPLE)

    text.clip = ("4", 1)
    checkpoint.ckpt_name = text.output["CONDITIONING"]

    assert workflow.dependents("4") == {"3", "6", "7", "8"}
    assert exported(workflow) == BASIC_API_EXAMPLE


def test_loaded_node_used_by_others_cannot_be_removed():
    workflow = loaded(BASIC_API_EXAMPLE)

    with pytest.raises(NodeHasDependentsError) as excinfo:
        workflow.remove_node("4")

    assert str(excinfo.value) == (
        "Cannot remove node '4': nodes '3', '6', '7' and '8' depend on it."
    )


def test_loaded_node_rejects_a_change_that_creates_a_cycle():
    workflow = loaded(
        {
            **BASIC_API_EXAMPLE,
            "10": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["8", 0], "vae": ["4", 2]},
            },
        }
    )

    with pytest.raises(InvalidNodeChangeError) as excinfo:
        workflow.update_node("3", latent_image=("10", 0))

    assert str(excinfo.value) == (
        "Cannot change 'latent_image' of node '3': node '10' depends on '3', "
        "so the change would create the cycle '3' -> '10' -> '8' -> '3'."
    )
    assert exported(workflow)["3"] == BASIC_API_EXAMPLE["3"]


def test_widget_takes_a_link_of_its_own_type_when_loaded():
    workflow = loaded(
        {
            **with_inputs("3", steps=["11", 0]),
            "11": {"class_type": "PrimitiveInt", "inputs": {"value": 30}},
        }
    )

    assert workflow.dependencies("3") == {"4", "5", "6", "7", "11"}


def test_loaded_node_can_be_rewired():
    workflow = loaded(BASIC_API_EXAMPLE)
    text = workflow.add_node(CLIPTextEncode, text="blurry", clip=("4", 1))

    workflow.update_node("3", negative=text.output["CONDITIONING"])

    assert workflow.dependencies("3") == {"4", "5", "6", text.node_id}
    assert workflow.dependents("7") == set()


def test_add_node_after_load_continues_the_numbering():
    built = Workflow()
    checkpoint = built.add_node(CheckpointLoaderSimple, ckpt_name="model.safetensors")
    built.add_node(CLIPTextEncode, text="a cat", clip=checkpoint.output["CLIP"])
    workflow = loaded(built.get_comfy_ui_workflow_format())

    node = workflow.add_node(
        CLIPTextEncode,
        text="a dog",
        clip=workflow.nodes["CheckpointLoaderSimple_0"].output["CLIP"],
    )

    assert node.node_id == "CLIPTextEncode_2"


def test_add_node_after_load_skips_the_ids_of_loaded_nodes():
    workflow = loaded(
        {
            "CheckpointLoaderSimple_1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "model.safetensors"},
            }
        }
    )

    node = workflow.add_node(CheckpointLoaderSimple, ckpt_name="other.safetensors")

    assert node.node_id == "CheckpointLoaderSimple_2"
    assert list(workflow.nodes) == [
        "CheckpointLoaderSimple_1",
        "CheckpointLoaderSimple_2",
    ]


def test_load_workflow_does_not_change_the_given_workflow():
    given = copy.deepcopy(BASIC_API_EXAMPLE)

    workflow = loaded(given)
    workflow.update_node("3", steps=30, positive=("7", 0), negative=("6", 0))

    assert given == BASIC_API_EXAMPLE


def test_meta_of_the_nodes_is_accepted_and_left_out():
    with_meta = {
        node_id: {**node, "_meta": {"title": node["class_type"]}}
        for node_id, node in BASIC_API_EXAMPLE.items()
    }

    assert exported(loaded(with_meta)) == BASIC_API_EXAMPLE


def test_wrapped_values_are_unwrapped():
    workflow = loaded(with_inputs("6", text={"__value__": "a cat"}))

    assert workflow.nodes["6"].text == "a cat"


def test_wrapped_lists_are_values_and_not_links():
    workflow = loaded(
        {
            "1": {
                "class_type": "RecordAudio",
                "inputs": {"audio": {"__value__": [0.25, 0.5]}},
            }
        }
    )

    assert workflow.nodes["1"].audio == [0.25, 0.5]
    assert workflow.dependencies("1") == set()
    assert exported(workflow)["1"]["inputs"]["audio"] == {"__value__": [0.25, 0.5]}


def test_inputs_are_read_by_their_comfy_ui_names():
    workflow = loaded(MOGE_RENDER)

    assert workflow.nodes["4"].output_ == "normal_opengl"
    assert exported(workflow)["4"]["inputs"] == MOGE_RENDER["4"]["inputs"]


def test_class_type_that_is_not_a_python_name_is_loaded():
    workflow = loaded(
        {
            "4": BASIC_API_EXAMPLE["4"],
            "10": {
                "class_type": "Epsilon Scaling",
                "inputs": {"model": ["4", 0], "scaling_factor": 1.005},
            },
        }
    )

    assert isinstance(workflow.nodes["10"], EpsilonScaling)
    assert workflow.dependencies("10") == {"4"}


class LocalNode(BaseNode):
    __slots__ = ()
    CATEGORY = "test"


def test_only_node_classes_of_the_library_are_loaded():
    with pytest.raises(InvalidWorkflowError) as excinfo:
        Workflow().load_workflow({"1": {"class_type": "LocalNode", "inputs": {}}})

    assert str(excinfo.value) == "Cannot load node '1': unknown class_type 'LocalNode'."


def test_node_classes_are_found_without_importing_them_first():
    code = (
        "from comfy_ui_api_workflow._impl.workflow import Workflow\n"
        "workflow = Workflow()\n"
        f"workflow.load_workflow({MOGE_RENDER!r})\n"
        "print(type(workflow.nodes['4']).__name__)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "MoGeRender"


def subclasses(node_type):
    for subclass in node_type.__subclasses__():
        yield subclass
        yield from subclasses(subclass)


def test_each_class_type_names_a_single_node_class_of_the_library():
    for module in pkgutil.walk_packages(
        nodes_package.__path__, f"{nodes_package.__name__}."
    ):
        importlib.import_module(module.name)
    class_types = [
        node_type.CLASS_TYPE
        for node_type in set(subclasses(BaseNode))
        if node_type.__module__.startswith(f"{nodes_package.__name__}.")
    ]

    assert len(class_types) > 600
    assert len(set(class_types)) == len(class_types)


@pytest.mark.skipif(not WAN_WORKFLOW.exists(), reason=f"{WAN_WORKFLOW.name} not found")
def test_exported_wan_workflow_is_loaded():
    exported_wan = json.loads(WAN_WORKFLOW.read_text(encoding="utf-8"))

    workflow = loaded(exported_wan)

    assert exported(workflow) == {
        node_id: {"class_type": node["class_type"], "inputs": node["inputs"]}
        for node_id, node in exported_wan.items()
    }
    assert {
        node_id: workflow.dependencies(node_id) for node_id in workflow.nodes
    } == linked_sources(exported_wan)


@pytest.mark.parametrize(
    ("workflow_format", "message"),
    [
        pytest.param(
            [],
            "Cannot load the workflow: expected a dict of nodes, got list.",
            id="workflow-not-dict",
        ),
        pytest.param(
            {
                "last_node_id": 9,
                "last_link_id": 9,
                "nodes": [{"id": 4, "type": "CheckpointLoaderSimple"}],
                "links": [[1, 4, 0, 3, 0, "MODEL"]],
                "version": 0.4,
            },
            "Cannot load the workflow: it is in the ComfyUI UI format; "
            "export it with 'Export (API)' and load that file.",
            id="ui-format",
        ),
        pytest.param(
            {3: BASIC_API_EXAMPLE["4"]},
            "Cannot load the workflow: node ids must be str, got 3.",
            id="id-not-str",
        ),
        pytest.param(
            {"4": "CheckpointLoaderSimple"},
            "Cannot load node '4': expected a dict with 'class_type' and 'inputs', "
            "got str.",
            id="node-not-dict",
        ),
        pytest.param(
            {"4": {"inputs": {"ckpt_name": "model.safetensors"}}},
            "Cannot load node '4': 'class_type' is missing.",
            id="no-class-type",
        ),
        pytest.param(
            {"4": {"class_type": 4, "inputs": {}}},
            "Cannot load node '4': 'class_type' must be a str, got int.",
            id="class-type-not-str",
        ),
        pytest.param(
            {"4": {"class_type": "NoSuchNode", "inputs": {}}},
            "Cannot load node '4': unknown class_type 'NoSuchNode'.",
            id="unknown-class-type",
        ),
        pytest.param(
            {"4": {"class_type": "CheckpointLoaderSimple"}},
            "Cannot load node '4': 'inputs' is missing.",
            id="no-inputs",
        ),
        pytest.param(
            {"4": {"class_type": "CheckpointLoaderSimple", "inputs": ["ckpt_name"]}},
            "Cannot load node '4': 'inputs' must be a dict, got list.",
            id="inputs-not-dict",
        ),
        pytest.param(
            with_inputs("4", foo=1),
            "Cannot load node '4': CheckpointLoaderSimple has no input 'foo'.",
            id="unknown-input",
        ),
        pytest.param(
            {
                **MOGE_RENDER,
                "4": {
                    "class_type": "MoGeRender",
                    "inputs": {"moge_geometry": ["3", 0], "output_": "depth"},
                },
            },
            "Cannot load node '4': MoGeRender has no input 'output_'.",
            id="python-name-of-renamed-input",
        ),
        pytest.param(
            without_inputs("3", "model"),
            "Cannot load node '3': KSampler is missing the input 'model'.",
            id="missing-input",
        ),
        pytest.param(
            without_inputs("3", "model", "positive", "negative"),
            "Cannot load node '3': KSampler is missing the inputs 'model', "
            "'positive' and 'negative'.",
            id="missing-inputs",
        ),
        pytest.param(
            with_inputs("3", steps=0),
            "Cannot load node '3': steps must be in closed range: [1, 10000].",
            id="invalid-value",
        ),
        pytest.param(
            with_inputs("3", steps="20"),
            "Cannot load node '3': steps must be an int or a link, not str.",
            id="value-of-another-type",
        ),
        pytest.param(
            with_inputs("3", sampler_name={"__value__": ["euler"]}),
            "Cannot load node '3': sampler_name must be a str or a link, not list.",
            id="wrapped-list-for-a-combo",
        ),
        pytest.param(
            with_inputs("3", model="v1-5-pruned-emaonly.safetensors"),
            "Cannot load node '3': model must be a link, not str.",
            id="value-for-a-link-only-input",
        ),
        pytest.param(
            with_inputs("3", model=[4, 0]),
            "Cannot load node '3': input 'model' is a list but not a link "
            "[node_id, output_index].",
            id="link-id-not-str",
        ),
        pytest.param(
            with_inputs("3", model=["4"]),
            "Cannot load node '3': input 'model' is a list but not a link "
            "[node_id, output_index].",
            id="link-too-short",
        ),
        pytest.param(
            with_inputs("3", model=("4", 0, 1)),
            "Cannot load node '3': input 'model' is a list but not a link "
            "[node_id, output_index].",
            id="link-too-long",
        ),
        pytest.param(
            with_inputs("3", model=["99", 0]),
            "Cannot load node '3': input 'model' uses node '99', "
            "which is not in the workflow.",
            id="link-to-missing-node",
        ),
        pytest.param(
            with_inputs("6", clip=["4", 5]),
            "Cannot load node '6': input 'clip' uses output 5 of node '4', "
            "but CheckpointLoaderSimple has no output 5.",
            id="output-out-of-range",
        ),
        pytest.param(
            with_inputs("6", clip=["4", -1]),
            "Cannot load node '6': input 'clip' uses output -1 of node '4', "
            "but CheckpointLoaderSimple has no output -1.",
            id="negative-output",
        ),
        pytest.param(
            with_inputs("6", clip=["4", True]),
            "Cannot load node '6': input 'clip' uses output True of node '4', "
            "but CheckpointLoaderSimple has no output True.",
            id="bool-output",
        ),
        pytest.param(
            with_inputs("3", model=["6", 0]),
            "Cannot load node '3': input 'model' uses output 0 of node '6', "
            "which is CONDITIONING, but the input takes MODEL.",
            id="output-type-mismatch",
        ),
        pytest.param(
            with_inputs("3", steps=["4", 0]),
            "Cannot load node '3': input 'steps' uses output 0 of node '4', "
            "which is MODEL, but the input takes INT.",
            id="widget-link-type-mismatch",
        ),
        pytest.param(
            {
                **MOGE_RENDER,
                "4": {
                    "class_type": "MoGeRender",
                    "inputs": {"moge_geometry": ["3", 0], "output": ["2", 0]},
                },
            },
            "Cannot load node '4': input 'output' uses output 0 of node '2', "
            "which is IMAGE, but the input takes COMBO.",
            id="type-mismatch-on-a-renamed-input",
        ),
        pytest.param(
            {
                "1": {"class_type": "EmptyImage", "inputs": {"width": 8, "height": 8}},
                "2": {"class_type": "CreateVideo", "inputs": {"images": ["1", 0]}},
                "3": {
                    "class_type": "GetVideoComponents",
                    "inputs": {"video": ["2", 0]},
                },
                "4": {
                    "class_type": "ImageScale",
                    "inputs": {
                        "image": ["1", 0],
                        "upscale_method": ["3", 3],
                        "width": 8,
                        "height": 8,
                        "crop": "disabled",
                    },
                },
            },
            "Cannot load node '4': input 'upscale_method' uses output 3 of node '3', "
            "which is COMBO, but the input takes V1 COMBO.",
            id="combo-output-into-a-v1-combo",
        ),
        pytest.param(
            with_inputs("8", samples=["8", 0]),
            "Cannot load node '8': input 'samples' uses the node's own output.",
            id="own-output",
        ),
        pytest.param(
            {
                "9": BASIC_API_EXAMPLE["4"],
                "2": {
                    "class_type": "ComfySwitchNode",
                    "inputs": {"switch": True, "on_true": ["1", 0]},
                },
                "1": {
                    "class_type": "ModelMergeSD1",
                    "inputs": {
                        "model1": ["9", 0],
                        "model2": ["9", 0],
                        "time_embed.": ["2", 0],
                    },
                },
            },
            "Cannot load node '1': input 'time_embed.' uses node '2', which depends "
            "on '1', so the workflow has the cycle '1' -> '2' -> '1'.",
            id="cycle-through-a-renamed-input",
        ),
        pytest.param(
            IMG2IMG_LOOP,
            "Cannot load node '10': input 'pixels' uses node '8', which depends on "
            "'10', so the workflow has the cycle '10' -> '8' -> '3' -> '10'.",
            id="cycle",
        ),
    ],
)
def test_invalid_workflow_is_rejected_with_the_reason(workflow_format, message):
    with pytest.raises(InvalidWorkflowError) as excinfo:
        Workflow().load_workflow(workflow_format)

    assert str(excinfo.value) == message


@pytest.mark.parametrize(
    "workflow_format",
    [
        [],
        {**BASIC_API_EXAMPLE, "99": {"class_type": "NoSuchNode", "inputs": {}}},
        with_inputs("3", steps=0),
        with_inputs("9", images=["99", 0]),
        with_inputs("9", images=["8", 1]),
        IMG2IMG_LOOP,
    ],
    ids=[
        "workflow-not-dict",
        "last-node-unknown",
        "invalid-value",
        "link-to-missing-node",
        "output-out-of-range",
        "cycle",
    ],
)
def test_rejected_workflow_keeps_the_current_one(workflow_format):
    workflow = Workflow()
    checkpoint = workflow.add_node(
        CheckpointLoaderSimple, ckpt_name="model.safetensors"
    )
    text = workflow.add_node(
        CLIPTextEncode, text="a cat", clip=checkpoint.output["CLIP"]
    )
    before = state(workflow)

    with pytest.raises(InvalidWorkflowError):
        workflow.load_workflow(workflow_format)

    assert state(workflow) == before
    with pytest.raises(InvalidNodeChangeError):
        text.clip = text.output["CONDITIONING"]
    assert workflow.add_node(EmptyLatentImage).node_id == "EmptyLatentImage_2"


def test_rejected_value_keeps_the_node_error_as_cause():
    with pytest.raises(InvalidWorkflowError) as excinfo:
        Workflow().load_workflow(with_inputs("3", steps=0))

    assert isinstance(excinfo.value.__cause__, InvalidNodeParameterValue)


def test_rejects_a_node_using_the_same_link_for_positive_and_negative():
    with pytest.raises(InvalidWorkflowError) as excinfo:
        Workflow().load_workflow(with_inputs("3", negative=["6", 0]))

    assert str(excinfo.value) == (
        "Cannot load node '3': positive and negative must not use the same input."
    )


def test_rejected_cycle_keeps_the_graph_error_as_cause():
    with pytest.raises(InvalidWorkflowError) as excinfo:
        Workflow().load_workflow(IMG2IMG_LOOP)

    assert isinstance(excinfo.value.__cause__, CycleError)


def test_invalid_workflow_error_is_workflow_exception():
    assert issubclass(InvalidWorkflowError, WorkflowException)


@pytest.mark.parametrize(
    "clone",
    [lambda error: pickle.loads(pickle.dumps(error)), copy.copy],
    ids=["pickle", "copy"],
)
@pytest.mark.parametrize("node_id", ["3", None], ids=["node", "workflow"])
def test_invalid_workflow_error_survives_pickle_and_copy(clone, node_id):
    error = InvalidWorkflowError(node_id, "a reason")

    restored = clone(error)

    assert str(restored) == str(error)
    assert (restored.node_id, restored.reason) == (node_id, "a reason")
