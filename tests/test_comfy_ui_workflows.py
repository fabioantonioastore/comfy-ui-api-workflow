import json
from pathlib import Path

import pytest

from comfy_ui_api_workflow._impl.nodes.image import LoadImage, SaveImage
from comfy_ui_api_workflow._impl.nodes.model.conditioning import CLIPTextEncode
from comfy_ui_api_workflow._impl.nodes.model.conditioning.wan import WanImageToVideo
from comfy_ui_api_workflow._impl.nodes.model.latent import EmptyLatentImage, VAEDecode
from comfy_ui_api_workflow._impl.nodes.model.loaders import (
    CheckpointLoaderSimple,
    CLIPLoader,
    LoraLoaderModelOnly,
    UNETLoader,
    VAELoader,
)
from comfy_ui_api_workflow._impl.nodes.model.patch.stable_diffusion import (
    ModelSamplingSD3,
)
from comfy_ui_api_workflow._impl.nodes.model.sampling import KSampler, KSamplerAdvanced
from comfy_ui_api_workflow._impl.nodes.utilities import ComfyMathExpression
from comfy_ui_api_workflow._impl.nodes.utilities.logic import ComfySwitchNode
from comfy_ui_api_workflow._impl.nodes.utilities.primitive import (
    PrimitiveBoolean,
    PrimitiveFloat,
    PrimitiveInt,
)
from comfy_ui_api_workflow._impl.nodes.video import CreateVideo, SaveVideo

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


def attribute(cls, name):
    # Dotted ComfyUI names, like "values.a", map to attributes through INPUT_NAMES.
    return {comfy: attr for attr, comfy in cls.INPUT_NAMES.items()}.get(name, name)


def sent(*nodes):
    prompt = {}
    for node in nodes:
        prompt.update(node.get_comfy_ui_workflow_format())
    return json.loads(json.dumps(prompt))


def test_default_workflow_matches_the_comfy_ui_basic_api_example():
    checkpoint = CheckpointLoaderSimple(
        node_id="4", ckpt_name="v1-5-pruned-emaonly.safetensors"
    )
    latent = EmptyLatentImage(node_id="5", width=512, height=512, batch_size=1)
    positive = CLIPTextEncode(
        node_id="6",
        text="masterpiece best quality girl",
        clip=checkpoint.output["CLIP"],
    )
    negative = CLIPTextEncode(
        node_id="7", text="bad hands", clip=checkpoint.output["CLIP"]
    )
    sampler = KSampler(
        node_id="3",
        model=checkpoint.output["MODEL"],
        sampler_name="euler",
        scheduler="normal",
        positive=positive.output["CONDITIONING"],
        negative=negative.output["CONDITIONING"],
        latent_image=latent.output["LATENT"],
        seed=8566257,
        steps=20,
        cfg=8,
        denoise=1,
    )
    decode = VAEDecode(
        node_id="8", samples=sampler.output["LATENT"], vae=checkpoint.output["VAE"]
    )
    save = SaveImage(
        node_id="9", images=decode.output["IMAGE"], filename_prefix="ComfyUI"
    )

    assert (
        sent(sampler, checkpoint, latent, positive, negative, decode, save)
        == BASIC_API_EXAMPLE
    )


@pytest.mark.skipif(not WAN_WORKFLOW.exists(), reason=f"{WAN_WORKFLOW.name} not found")
def test_exported_wan_workflow_is_rebuilt_node_by_node():
    classes = {
        cls.CLASS_TYPE: cls
        for cls in (
            LoadImage,
            WanImageToVideo,
            CreateVideo,
            ModelSamplingSD3,
            CLIPTextEncode,
            PrimitiveInt,
            PrimitiveFloat,
            PrimitiveBoolean,
            VAEDecode,
            VAELoader,
            CLIPLoader,
            UNETLoader,
            LoraLoaderModelOnly,
            KSamplerAdvanced,
            ComfySwitchNode,
            SaveVideo,
            ComfyMathExpression,
        )
    }
    exported = json.loads(WAN_WORKFLOW.read_text(encoding="utf-8"))
    expected = {
        node_id: {"class_type": node["class_type"], "inputs": node["inputs"]}
        for node_id, node in exported.items()
    }

    rebuilt = [
        classes[node["class_type"]](
            node_id=node_id,
            **{
                attribute(classes[node["class_type"]], name): (
                    tuple(value) if isinstance(value, list) else value
                )
                for name, value in node["inputs"].items()
            },
        )
        for node_id, node in exported.items()
    ]

    assert sent(*rebuilt) == expected
