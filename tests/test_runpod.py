import base64
import json
import struct
import zlib

import pytest

from comfy_ui_api_workflow._impl.errors.workflow import InvalidWorkflowError
from comfy_ui_api_workflow._impl.nodes.image import EmptyImage, SaveImage
from comfy_ui_api_workflow._impl.nodes.model.conditioning import CLIPTextEncode
from comfy_ui_api_workflow._impl.nodes.model.latent import EmptyLatentImage, VAEDecode
from comfy_ui_api_workflow._impl.nodes.model.loaders import CheckpointLoaderSimple
from comfy_ui_api_workflow._impl.nodes.model.sampling import KSampler
from comfy_ui_api_workflow._impl.nodes.utilities import ComfyMathExpression
from comfy_ui_api_workflow._impl.nodes.utilities.primitive import PrimitiveInt
from comfy_ui_api_workflow._impl.nodes.video import CreateVideo, SaveVideo
from comfy_ui_api_workflow._impl.pipeline import Pipeline
from comfy_ui_api_workflow._impl.workflow import Workflow

# Every test here sends a real job to the endpoint in .env, so they only run with
# `hatch test -- --runpod`. The checkpoint is the one on the endpoint's volume.
pytestmark = pytest.mark.runpod
CHECKPOINT = "sd_xl_base_1.0.safetensors"


class TextToImage(Pipeline):
    # SDXL was trained at 1024x1024; with fewer steps or a smaller size it only
    # produces blurred shapes.
    def build(self, prompt: str, size: int = 1024, steps: int = 25):
        model = self.add_node(CheckpointLoaderSimple, ckpt_name=CHECKPOINT)
        positive = self.add_node(CLIPTextEncode, text=prompt, clip=model.output["CLIP"])
        negative = self.add_node(
            CLIPTextEncode,
            text="blurry, low quality, distorted, deformed",
            clip=model.output["CLIP"],
        )
        latent = self.add_node(EmptyLatentImage, width=size, height=size, batch_size=1)
        sampler = self.add_node(
            KSampler,
            model=model.output["MODEL"],
            sampler_name="dpmpp_2m",
            scheduler="karras",
            positive=positive.output["CONDITIONING"],
            negative=negative.output["CONDITIONING"],
            latent_image=latent.output["LATENT"],
            seed=42,
            steps=steps,
            cfg=7.0,
        )
        image = self.add_node(
            VAEDecode, samples=sampler.output["LATENT"], vae=model.output["VAE"]
        )
        return {"IMAGE": image.output["IMAGE"]}


def sent(workflow):
    return json.loads(json.dumps(workflow.get_comfy_ui_workflow_format()))


def images_of(job):
    assert job["status"] == "COMPLETED", job.get("error")
    return [base64.b64decode(image["data"]) for image in job["output"]["images"]]


def png_size(png):
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", png[16:24])


def first_pixel(png):
    # The first pixel of the first row is stored as is, whatever the row filter.
    data, position = b"", 8
    while position < len(png):
        (length,) = struct.unpack(">I", png[position : position + 4])
        if png[position + 4 : position + 8] == b"IDAT":
            data += png[position + 8 : position + 8 + length]
        position += 12 + length
    return tuple(zlib.decompress(data)[1:4])


def solid_image(workflow, **inputs):
    image = workflow.add_node(EmptyImage, **{"batch_size": 1, **inputs})
    workflow.add_node(SaveImage, images=image.output["IMAGE"], filename_prefix="test")
    return image


def test_workflow_built_with_the_library_runs_on_comfy_ui(runpod):
    workflow = Workflow()
    solid_image(workflow, width=64, height=32, color=0xFF0000)

    (png,) = images_of(runpod.run(sent(workflow)))

    assert png_size(png) == (64, 32)
    assert first_pixel(png) == (255, 0, 0)


def test_loaded_workflow_runs_on_comfy_ui(runpod):
    built = Workflow()
    solid_image(built, width=48, height=16, color=0x00FF00)
    workflow = Workflow()

    workflow.load_workflow(json.loads(json.dumps(built.get_comfy_ui_workflow_format())))
    (png,) = images_of(runpod.run(sent(workflow)))

    assert png_size(png) == (48, 16)
    assert first_pixel(png) == (0, 255, 0)


def test_math_expression_values_reach_comfy_ui(runpod):
    workflow = Workflow()
    number = workflow.add_node(PrimitiveInt, value=16)
    math = workflow.add_node(
        ComfyMathExpression, expression="a * 4", values_a=number.output["INT"]
    )
    solid_image(workflow, width=math.output["INT"], height=8, color=0)

    (png,) = images_of(runpod.run(sent(workflow)))

    assert png_size(png) == (64, 8)


def test_save_video_branch_reaches_comfy_ui(runpod):
    workflow = Workflow()
    frames = workflow.add_node(
        EmptyImage, width=128, height=128, batch_size=16, color=0x2060FF
    )
    video = workflow.add_node(CreateVideo, images=frames.output["IMAGE"], fps=8.0)
    workflow.add_node(
        SaveVideo,
        video=video.output["VIDEO"],
        filename_prefix="test",
        format="mp4",
        format_codec="av1",
        format_codec_encoding="re-encode",
        format_codec_encoding_crf=40.0,
    )

    job = runpod.run(sent(workflow))
    (clip,) = images_of(job)

    # AV1 is not the mp4 default (h264), so it shows that "format.codec" took effect.
    assert job["output"]["images"][0]["filename"].endswith(".mp4")
    assert b"ftyp" in clip and b"av01" in clip and b"avc1" not in clip


def test_text_to_image_pipeline_generates_with_sdxl(runpod):
    workflow = Workflow()
    pipeline = workflow.add_pipeline(
        TextToImage,
        prompt="a red apple on a wooden table, studio photo, soft light, detailed",
    )
    workflow.add_node(
        SaveImage, images=pipeline.output["IMAGE"], filename_prefix="test"
    )

    (png,) = images_of(runpod.run(sent(workflow), timeout=900))

    assert png_size(png) == (1024, 1024)


def test_link_the_library_refuses_is_refused_by_comfy_ui_too(runpod):
    float_into_int = {
        "1": {"class_type": "PrimitiveFloat", "inputs": {"value": 64.0}},
        "2": {
            "class_type": "EmptyImage",
            "inputs": {"width": ["1", 0], "height": 64, "batch_size": 1, "color": 0},
        },
        "3": {
            "class_type": "SaveImage",
            "inputs": {"images": ["2", 0], "filename_prefix": "test"},
        },
    }

    with pytest.raises(InvalidWorkflowError) as excinfo:
        Workflow().load_workflow(float_into_int)
    job = runpod.run(float_into_int)

    assert "which is FLOAT, but the input takes INT" in str(excinfo.value)
    assert job["status"] == "FAILED"
    assert "received_type(FLOAT) mismatch input_type(INT)" in job["error"]


def test_combo_output_into_a_v1_combo_is_refused_by_both(runpod):
    # A V1 combo's type is its list of options, so ComfyUI links no COMBO output to it.
    combo_into_v1_combo = {
        "1": {
            "class_type": "EmptyImage",
            "inputs": {"width": 8, "height": 8, "batch_size": 1, "color": 0},
        },
        "2": {"class_type": "CreateVideo", "inputs": {"images": ["1", 0], "fps": 1.0}},
        "3": {"class_type": "GetVideoComponents", "inputs": {"video": ["2", 0]}},
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
        "5": {
            "class_type": "SaveImage",
            "inputs": {"images": ["4", 0], "filename_prefix": "test"},
        },
    }

    with pytest.raises(InvalidWorkflowError) as excinfo:
        Workflow().load_workflow(combo_into_v1_combo)
    job = runpod.run(combo_into_v1_combo)

    assert "which is COMBO, but the input takes V1 COMBO" in str(excinfo.value)
    assert job["status"] == "FAILED"
    assert "received_type(COMBO) mismatch input_type([" in job["error"]


def test_comfy_ui_runs_the_shared_conditioning_the_library_refuses(runpod):
    # ComfyUI has no such rule: only the library refuses it.
    shared_conditioning = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CHECKPOINT},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "a red apple on a wooden table", "clip": ["1", 1]},
        },
        "3": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
        },
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["2", 0],
                "latent_image": ["3", 0],
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed": 42,
                "steps": 4,
                "cfg": 7.0,
                "denoise": 1.0,
            },
        },
        "5": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["4", 0], "vae": ["1", 2]},
        },
        "6": {
            "class_type": "SaveImage",
            "inputs": {"images": ["5", 0], "filename_prefix": "test"},
        },
    }

    with pytest.raises(InvalidWorkflowError) as excinfo:
        Workflow().load_workflow(shared_conditioning)
    (png,) = images_of(runpod.run(shared_conditioning))

    assert str(excinfo.value) == (
        "Cannot load node '4': positive and negative must not use the same input."
    )
    assert png_size(png) == (512, 512)
