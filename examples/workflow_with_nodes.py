import json

from comfy_ui_api_workflow._impl.nodes.image import SaveImage
from comfy_ui_api_workflow._impl.nodes.model.conditioning import CLIPTextEncode
from comfy_ui_api_workflow._impl.nodes.model.latent import EmptyLatentImage, VAEDecode
from comfy_ui_api_workflow._impl.nodes.model.loaders import CheckpointLoaderSimple
from comfy_ui_api_workflow._impl.nodes.model.sampling import KSampler
from comfy_ui_api_workflow._impl.workflow import Workflow


def build() -> Workflow:
    workflow = Workflow()

    model = workflow.add_node(
        CheckpointLoaderSimple, ckpt_name="sd_xl_base_1.0.safetensors"
    )
    positive = workflow.add_node(
        CLIPTextEncode,
        text="a red apple on a wooden table, studio photo, soft light",
        clip=model.output["CLIP"],
    )
    negative = workflow.add_node(
        CLIPTextEncode, text="blurry, low quality", clip=model.output["CLIP"]
    )
    latent = workflow.add_node(EmptyLatentImage, width=1024, height=1024)
    sampler = workflow.add_node(
        KSampler,
        model=model.output["MODEL"],
        sampler_name="dpmpp_2m",
        scheduler="karras",
        positive=positive.output["CONDITIONING"],
        negative=negative.output["CONDITIONING"],
        latent_image=latent.output["LATENT"],
        seed=42,
        steps=25,
        cfg=7.0,
    )
    image = workflow.add_node(
        VAEDecode, samples=sampler.output["LATENT"], vae=model.output["VAE"]
    )
    workflow.add_node(SaveImage, images=image.output["IMAGE"], filename_prefix="apple")

    # Inputs can still change after a node is added, with the same checks.
    workflow.update_node(sampler.node_id, steps=30)
    return workflow


if __name__ == "__main__":
    print(json.dumps(build().get_comfy_ui_workflow_format(), indent=2))
