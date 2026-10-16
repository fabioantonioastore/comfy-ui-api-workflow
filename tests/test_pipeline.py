import copy
import pickle

import pytest

from comfy_ui_api_workflow._impl.errors.data_structure import CycleError
from comfy_ui_api_workflow._impl.errors.node import (
    InvalidLinkError,
    InvalidNodeChangeError,
    InvalidNodeParameterValue,
    NodeHasDependentsError,
)
from comfy_ui_api_workflow._impl.errors.output import OutputNotFoundError
from comfy_ui_api_workflow._impl.errors.pipeline import PipelineError
from comfy_ui_api_workflow._impl.exceptions import PipelineException
from comfy_ui_api_workflow._impl.nodes.image import LoadImage, SaveImage
from comfy_ui_api_workflow._impl.nodes.image.upscaling import ImageScaleBy
from comfy_ui_api_workflow._impl.nodes.model.conditioning import CLIPTextEncode
from comfy_ui_api_workflow._impl.nodes.model.latent import EmptyLatentImage, VAEDecode
from comfy_ui_api_workflow._impl.nodes.model.loaders import CheckpointLoaderSimple
from comfy_ui_api_workflow._impl.nodes.model.sampling import KSampler
from comfy_ui_api_workflow._impl.nodes.utilities.logic import ComfySwitchNode
from comfy_ui_api_workflow._impl.pipeline import Pipeline
from comfy_ui_api_workflow._impl.workflow import Workflow


class TextToImage(Pipeline):
    def build(
        self, prompt: str, checkpoint: str, steps: int = 20, negative: str = "blurry"
    ):
        model = self.add_node(CheckpointLoaderSimple, ckpt_name=checkpoint)
        positive = self.add_node(CLIPTextEncode, text=prompt, clip=model.output["CLIP"])
        unwanted = self.add_node(
            CLIPTextEncode, text=negative, clip=model.output["CLIP"]
        )
        latent = self.add_node(EmptyLatentImage, width=512, height=512, batch_size=1)
        sampler = self.add_node(
            KSampler,
            model=model.output["MODEL"],
            sampler_name="euler",
            scheduler="normal",
            positive=positive.output["CONDITIONING"],
            negative=unwanted.output["CONDITIONING"],
            latent_image=latent.output["LATENT"],
            steps=steps,
        )
        image = self.add_node(
            VAEDecode, samples=sampler.output["LATENT"], vae=model.output["VAE"]
        )
        return {"IMAGE": image.output["IMAGE"]}


class Upscale(Pipeline):
    def build(self, image: tuple[str, int], scale: float = 2.0):
        scaled = self.add_node(
            ImageScaleBy, image=image, upscale_method="lanczos", scale_by=scale
        )
        return {"IMAGE": scaled.output["IMAGE"]}


class SaveOnly(Pipeline):
    def build(self, image: tuple[str, int]):
        self.add_node(SaveImage, images=image, filename_prefix="pipeline")


class AnyParameters(Pipeline):
    def build(self, **parameters):
        latent = self.add_node(EmptyLatentImage, **parameters)
        return {"LATENT": latent.output["LATENT"]}


class WrongLink(Pipeline):
    def build(self):
        model = self.add_node(CheckpointLoaderSimple, ckpt_name="model.safetensors")
        text = self.add_node(CLIPTextEncode, text="a cat", clip=model.output["CLIP"])
        self.add_node(
            VAEDecode, samples=text.output["CONDITIONING"], vae=model.output["VAE"]
        )


class OwnOutput(Pipeline):
    def build(self):
        model = self.add_node(CheckpointLoaderSimple, ckpt_name="model.safetensors")
        index = int(model.node_id.rsplit("_", 1)[1]) + 1
        self.add_node(CLIPTextEncode, text="a cat", clip=(f"CLIPTextEncode_{index}", 0))


class MissingSource(Pipeline):
    def build(self):
        self.add_node(CLIPTextEncode, text="a cat", clip=("Missing_9", 1))


class FailingBuild(Pipeline):
    def build(self, message: str = "no such style"):
        self.add_node(CheckpointLoaderSimple, ckpt_name="model.safetensors")
        raise ValueError(message)


class SwallowedRejection(Pipeline):
    def build(self):
        model = self.add_node(CheckpointLoaderSimple, ckpt_name="model.safetensors")
        try:
            self.add_node(
                VAEDecode, samples=model.output["MODEL"], vae=model.output["VAE"]
            )
        except InvalidLinkError:
            pass
        return {"MODEL": model.output["MODEL"]}


class RewiredCycle(Pipeline):
    def build(self):
        first = self.add_node(ComfySwitchNode, switch=True)
        second = self.add_node(
            ComfySwitchNode, switch=True, on_true=first.output["output"]
        )
        first.on_true = second.output["output"]


class RewiredToMissingNode(Pipeline):
    def build(self):
        switch = self.add_node(ComfySwitchNode, switch=True)
        switch.on_true = ("Missing_9", 0)


class SwallowedTwice(Pipeline):
    def build(self):
        model = self.add_node(CheckpointLoaderSimple, ckpt_name="model.safetensors")
        for source in (model.output["MODEL"], model.output["CLIP"]):
            try:
                self.add_node(VAEDecode, samples=source, vae=model.output["VAE"])
            except InvalidLinkError:
                pass


class BadOutput(Pipeline):
    def build(self, output):
        model = self.add_node(CheckpointLoaderSimple, ckpt_name="model.safetensors")
        return output(model)


def state(workflow):
    return (
        dict(workflow.nodes),
        workflow.get_comfy_ui_workflow_format(),
        {
            node_id: (workflow.dependencies(node_id), workflow.dependents(node_id))
            for node_id in workflow.nodes
        },
    )


def text_to_image(workflow, **parameters):
    return workflow.add_pipeline(
        TextToImage,
        **{"prompt": "a cat", "checkpoint": "model.safetensors", **parameters},
    )


def test_add_pipeline_adds_its_nodes_in_the_order_they_were_built():
    workflow = Workflow()

    text_to_image(workflow)

    assert {node_id: type(node) for node_id, node in workflow.nodes.items()} == {
        "CheckpointLoaderSimple_0": CheckpointLoaderSimple,
        "CLIPTextEncode_1": CLIPTextEncode,
        "CLIPTextEncode_2": CLIPTextEncode,
        "EmptyLatentImage_3": EmptyLatentImage,
        "KSampler_4": KSampler,
        "VAEDecode_5": VAEDecode,
    }
    assert list(workflow.nodes)[-1] == "VAEDecode_5"


def test_pipeline_nodes_get_the_parameters_and_links_of_the_build():
    workflow = Workflow()

    text_to_image(workflow, prompt="a dog", steps=30)

    assert workflow.dependencies("KSampler_4") == {
        "CheckpointLoaderSimple_0",
        "CLIPTextEncode_1",
        "CLIPTextEncode_2",
        "EmptyLatentImage_3",
    }
    assert workflow.nodes["CLIPTextEncode_1"].text == "a dog"
    assert workflow.nodes["CLIPTextEncode_2"].text == "blurry"
    assert workflow.nodes["KSampler_4"].steps == 30


def test_add_pipeline_returns_the_pipeline_with_its_nodes_and_outputs():
    workflow = Workflow()

    pipeline = text_to_image(workflow)

    assert isinstance(pipeline, TextToImage)
    assert list(pipeline.nodes) == list(workflow.nodes)
    assert all(
        node is workflow.nodes[node_id] for node_id, node in pipeline.nodes.items()
    )
    assert pipeline.output["IMAGE"] == workflow.nodes["VAEDecode_5"].output["IMAGE"]


def test_pipeline_uses_outputs_of_existing_nodes_as_parameters():
    workflow = Workflow()
    photo = workflow.add_node(LoadImage, image="photo.png")

    pipeline = workflow.add_pipeline(Upscale, image=photo.output["IMAGE"], scale=1.5)

    assert list(pipeline.nodes) == ["ImageScaleBy_1"]
    assert workflow.dependencies("ImageScaleBy_1") == {photo.node_id}
    assert workflow.nodes["ImageScaleBy_1"].scale_by == 1.5


def test_pipeline_outputs_feed_nodes_and_other_pipelines():
    workflow = Workflow()
    image = text_to_image(workflow).output["IMAGE"]

    upscaled = workflow.add_pipeline(Upscale, image=image).output["IMAGE"]
    save = workflow.add_node(SaveImage, images=upscaled)

    assert workflow.dependencies("ImageScaleBy_6") == {"VAEDecode_5"}
    assert workflow.dependencies(save.node_id) == {"ImageScaleBy_6"}


def test_pipeline_can_be_added_more_than_once():
    workflow = Workflow()

    first = text_to_image(workflow, prompt="a cat")
    second = text_to_image(workflow, prompt="a dog")

    assert len(workflow.nodes) == 12
    assert set(first.nodes).isdisjoint(second.nodes)
    assert second.output["IMAGE"] == ("VAEDecode_11", 0)


class OptionalImage(Pipeline):
    def build(self, image: tuple[str, int] | None = None):
        if image is None:
            image = self.add_node(LoadImage, image="photo.png").output["IMAGE"]
        self.add_node(SaveImage, images=image, filename_prefix="pipeline")


def test_optional_link_parameter_takes_none():
    workflow = Workflow()

    pipeline = workflow.add_pipeline(OptionalImage, image=None)

    assert [type(node).__name__ for node in pipeline.nodes.values()] == [
        "LoadImage",
        "SaveImage",
    ]


def test_pipeline_whose_build_takes_any_parameter():
    workflow = Workflow()

    pipeline = workflow.add_pipeline(AnyParameters, width=768, height=512)

    assert workflow.nodes["EmptyLatentImage_0"].width == 768
    assert pipeline.output["LATENT"] == ("EmptyLatentImage_0", 0)


def test_pipeline_without_outputs_still_adds_its_nodes():
    workflow = Workflow()
    photo = workflow.add_node(LoadImage, image="photo.png")

    pipeline = workflow.add_pipeline(SaveOnly, image=photo.output["IMAGE"])

    assert list(pipeline.nodes) == ["SaveImage_1"]
    with pytest.raises(OutputNotFoundError) as excinfo:
        pipeline.output["IMAGE"]
    assert str(excinfo.value) == "Pipeline 'SaveOnly' has no output 'IMAGE'."


def test_unknown_output_of_a_pipeline_lists_its_outputs():
    pipeline = text_to_image(Workflow())

    with pytest.raises(OutputNotFoundError) as excinfo:
        pipeline.output["LATENT"]

    assert str(excinfo.value) == (
        "Pipeline 'TextToImage' has no output 'LATENT'. Its outputs are: 'IMAGE'."
    )


def test_pipeline_nodes_are_protected_like_any_node():
    workflow = Workflow()
    pipeline = text_to_image(workflow)
    text = pipeline.nodes["CLIPTextEncode_1"]

    with pytest.raises(NodeHasDependentsError):
        workflow.remove_node("CheckpointLoaderSimple_0")
    with pytest.raises(InvalidNodeChangeError):
        workflow.update_node("VAEDecode_5", samples=text.output["CONDITIONING"])


REJECTED = [
    pytest.param(
        TextToImage,
        {"prompt": "a cat", "checkpoint": "model.safetensors", "stepz": 30},
        "Cannot add pipeline 'TextToImage': it has no parameter 'stepz' (it takes "
        "'prompt', 'checkpoint', 'steps' and 'negative'). "
        "No node of the pipeline was added.",
        id="unknown-parameter",
    ),
    pytest.param(
        SaveOnly,
        {"images": ("LoadImage_0", 0)},
        "Cannot add pipeline 'SaveOnly': it has no parameter 'images' (it takes "
        "'image'). No node of the pipeline was added.",
        id="unknown-parameter-of-a-single-parameter-pipeline",
    ),
    pytest.param(
        WrongLink,
        {"model": "model.safetensors"},
        "Cannot add pipeline 'WrongLink': it has no parameter 'model' (it takes no "
        "parameters). No node of the pipeline was added.",
        id="parameter-for-a-pipeline-without-parameters",
    ),
    pytest.param(
        TextToImage,
        {},
        "Cannot add pipeline 'TextToImage': it is missing the parameters 'prompt' "
        "and 'checkpoint'. No node of the pipeline was added.",
        id="missing-parameters",
    ),
    pytest.param(
        TextToImage,
        {"prompt": "a cat"},
        "Cannot add pipeline 'TextToImage': it is missing the parameter "
        "'checkpoint'. No node of the pipeline was added.",
        id="missing-parameter",
    ),
    pytest.param(
        TextToImage,
        {"prompt": "a cat", "checkpoint": "model.safetensors", "steps": "30"},
        "Cannot add pipeline 'TextToImage': steps must be an int, not str. "
        "No node of the pipeline was added.",
        id="parameter-of-another-type",
    ),
    pytest.param(
        Upscale,
        {"image": "photo.png"},
        "Cannot add pipeline 'Upscale': image must be a link, not str. "
        "No node of the pipeline was added.",
        id="value-for-a-link-parameter",
    ),
    pytest.param(
        Upscale,
        {"image": ("Missing_9", 0)},
        "Cannot add pipeline 'Upscale': parameter 'image' uses node 'Missing_9', "
        "which is not in the workflow. No node of the pipeline was added.",
        id="parameter-link-to-missing-node",
    ),
    pytest.param(
        Upscale,
        {"image": ("LoadImage_0", 3)},
        "Cannot add pipeline 'Upscale': parameter 'image' uses output 3 of node "
        "'LoadImage_0', but LoadImage has no output 3. "
        "No node of the pipeline was added.",
        id="parameter-link-to-missing-output",
    ),
    pytest.param(
        TextToImage,
        {"prompt": "a cat", "checkpoint": "model.safetensors", "steps": 0},
        "Cannot add pipeline 'TextToImage': node 5 of the pipeline (KSampler) was "
        "rejected: steps must be in closed range: [1, 10000]. "
        "No node of the pipeline was added.",
        id="invalid-value",
    ),
    pytest.param(
        WrongLink,
        {},
        "Cannot add pipeline 'WrongLink': node 3 of the pipeline (VAEDecode) was "
        "rejected: Cannot add node 'VAEDecode_3': input 'samples' uses output 0 of "
        "node 'CLIPTextEncode_2', which is CONDITIONING, but the input takes LATENT. "
        "No node of the pipeline was added.",
        id="incompatible-link",
    ),
    pytest.param(
        OwnOutput,
        {},
        "Cannot add pipeline 'OwnOutput': node 2 of the pipeline (CLIPTextEncode) was "
        "rejected: Cannot add edge 'CLIPTextEncode_2' -> 'CLIPTextEncode_2': a node "
        "cannot have an edge to itself. No node of the pipeline was added.",
        id="cycle",
    ),
    pytest.param(
        RewiredCycle,
        {},
        "Cannot add pipeline 'RewiredCycle': a change to node 1 of the pipeline "
        "(ComfySwitchNode) was rejected: Cannot change 'on_true' of node "
        "'ComfySwitchNode_1': node 'ComfySwitchNode_2' depends on "
        "'ComfySwitchNode_1', so the change would create the cycle "
        "'ComfySwitchNode_1' -> 'ComfySwitchNode_2' -> 'ComfySwitchNode_1'. "
        "No node of the pipeline was added.",
        id="cycle-made-by-a-change-in-build",
    ),
    pytest.param(
        RewiredToMissingNode,
        {},
        "Cannot add pipeline 'RewiredToMissingNode': a change to node 1 of the "
        "pipeline (ComfySwitchNode) was rejected: Cannot change 'on_true' of node "
        "'ComfySwitchNode_1': node 'Missing_9' is not in the workflow. "
        "No node of the pipeline was added.",
        id="missing-node-linked-by-a-change-in-build",
    ),
    pytest.param(
        MissingSource,
        {},
        "Cannot add pipeline 'MissingSource': node 1 of the pipeline (CLIPTextEncode) "
        "was rejected: node: Missing_9 not found. No node of the pipeline was added.",
        id="link-to-missing-node",
    ),
    pytest.param(
        SwallowedRejection,
        {},
        "Cannot add pipeline 'SwallowedRejection': node 2 of the pipeline (VAEDecode) "
        "was rejected: Cannot add node 'VAEDecode_2': input 'samples' uses output 0 "
        "of node 'CheckpointLoaderSimple_1', which is MODEL, but the input takes "
        "LATENT. No node of the pipeline was added.",
        id="rejection-caught-by-the-build",
    ),
    pytest.param(
        SwallowedTwice,
        {},
        "Cannot add pipeline 'SwallowedTwice': node 2 of the pipeline (VAEDecode) "
        "was rejected: Cannot add node 'VAEDecode_2': input 'samples' uses output 0 "
        "of node 'CheckpointLoaderSimple_1', which is MODEL, but the input takes "
        "LATENT. No node of the pipeline was added.",
        id="first-of-two-caught-rejections",
    ),
    pytest.param(
        FailingBuild,
        {},
        "Cannot add pipeline 'FailingBuild': its build failed with ValueError: no "
        "such style. No node of the pipeline was added.",
        id="build-error",
    ),
    pytest.param(
        FailingBuild,
        {"message": ""},
        "Cannot add pipeline 'FailingBuild': its build failed with ValueError. "
        "No node of the pipeline was added.",
        id="build-error-without-message",
    ),
    pytest.param(
        BadOutput,
        {"output": lambda model: {"MODEL": model}},
        "Cannot add pipeline 'BadOutput': its output 'MODEL' must be a link "
        "(node_id, output_index), not CheckpointLoaderSimple. "
        "No node of the pipeline was added.",
        id="output-not-a-link",
    ),
    pytest.param(
        BadOutput,
        {"output": lambda model: {"MODEL": (model.node_id, 3)}},
        "Cannot add pipeline 'BadOutput': its output 'MODEL' uses output 3 of node "
        "'CheckpointLoaderSimple_1', but CheckpointLoaderSimple has no output 3. "
        "No node of the pipeline was added.",
        id="output-the-node-does-not-have",
    ),
    pytest.param(
        BadOutput,
        {"output": lambda model: {"MODEL": ("Missing_9", 0)}},
        "Cannot add pipeline 'BadOutput': its output 'MODEL' uses node 'Missing_9', "
        "which is not in the workflow. No node of the pipeline was added.",
        id="output-of-a-missing-node",
    ),
    pytest.param(
        BadOutput,
        {"output": lambda model: {5: model.output["MODEL"]}},
        "Cannot add pipeline 'BadOutput': its output names must be str, not int. "
        "No node of the pipeline was added.",
        id="output-name-not-str",
    ),
    pytest.param(
        BadOutput,
        {"output": lambda model: model},
        "Cannot add pipeline 'BadOutput': its build must return a dict of outputs "
        "or None, not CheckpointLoaderSimple. No node of the pipeline was added.",
        id="build-result-not-a-dict",
    ),
    pytest.param(
        KSampler,
        {},
        "Cannot add pipeline 'KSampler': add_pipeline takes a Pipeline subclass. "
        "No node of the pipeline was added.",
        id="not-a-pipeline",
    ),
    pytest.param(
        TextToImage(),
        {},
        "Cannot add pipeline 'TextToImage': add_pipeline takes the pipeline class, "
        "not an instance. No node of the pipeline was added.",
        id="pipeline-instance",
    ),
    pytest.param(
        Pipeline,
        {},
        "Cannot add pipeline 'Pipeline': it does not define build. "
        "No node of the pipeline was added.",
        id="pipeline-without-build",
    ),
]


@pytest.mark.parametrize(("pipeline_type", "parameters", "message"), REJECTED)
def test_rejected_pipeline_adds_no_node_and_says_why(
    pipeline_type, parameters, message
):
    workflow = Workflow()
    workflow.add_node(LoadImage, image="photo.png")
    before = state(workflow)

    with pytest.raises(PipelineError) as excinfo:
        workflow.add_pipeline(pipeline_type, **parameters)

    assert str(excinfo.value) == message
    assert state(workflow) == before
    assert workflow.add_node(EmptyLatentImage).node_id == "EmptyLatentImage_1"


@pytest.mark.parametrize(
    ("pipeline_type", "parameters", "cause"),
    [
        (TextToImage, {"prompt": "a cat", "checkpoint": "m", "steps": "30"}, None),
        (TextToImage, {"prompt": "a cat", "checkpoint": "m", "steps": 0}, None),
        (WrongLink, {}, InvalidLinkError),
        (OwnOutput, {}, CycleError),
        (FailingBuild, {}, ValueError),
    ],
    ids=["parameter-type", "node-value", "node-link", "node-cycle", "build-error"],
)
def test_rejected_pipeline_keeps_the_original_error_as_cause(
    pipeline_type, parameters, cause
):
    with pytest.raises(PipelineError) as excinfo:
        Workflow().add_pipeline(pipeline_type, **parameters)

    assert isinstance(excinfo.value.__cause__, cause or InvalidNodeParameterValue)


def test_pipeline_cannot_add_nodes_outside_add_pipeline():
    with pytest.raises(PipelineError) as excinfo:
        TextToImage().build(prompt="a cat", checkpoint="model.safetensors")

    assert str(excinfo.value) == (
        "Cannot add pipeline 'TextToImage': add_node only works inside build, while "
        "Workflow.add_pipeline adds the pipeline. No node of the pipeline was added."
    )


def test_pipeline_cannot_add_nodes_after_it_was_added():
    workflow = Workflow()
    pipeline = text_to_image(workflow)

    with pytest.raises(PipelineError):
        pipeline.add_node(EmptyLatentImage)

    assert len(workflow.nodes) == 6


def test_node_used_by_a_pipeline_cannot_be_removed():
    workflow = Workflow()
    photo = workflow.add_node(LoadImage, image="photo.png")
    workflow.add_pipeline(Upscale, image=photo.output["IMAGE"])

    with pytest.raises(NodeHasDependentsError) as excinfo:
        workflow.remove_node(photo.node_id)

    assert str(excinfo.value) == (
        "Cannot remove node 'LoadImage_0': node 'ImageScaleBy_1' depends on it."
    )


def test_pipeline_error_is_pipeline_exception():
    assert issubclass(PipelineError, PipelineException)


@pytest.mark.parametrize(
    "clone",
    [lambda error: pickle.loads(pickle.dumps(error)), copy.copy],
    ids=["pickle", "copy"],
)
def test_pipeline_error_survives_pickle_and_copy(clone):
    error = PipelineError("TextToImage", "a reason")

    restored = clone(error)

    assert str(restored) == str(error)
    assert (restored.pipeline, restored.reason) == ("TextToImage", "a reason")


def test_base_pipeline_build_must_be_defined_by_a_subclass():
    with pytest.raises(NotImplementedError):
        Pipeline().build()
