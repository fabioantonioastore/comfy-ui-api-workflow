import json

from comfy_ui_api_workflow._impl.nodes.image import SaveImage
from comfy_ui_api_workflow._impl.nodes.image.batch import RepeatImageBatch
from comfy_ui_api_workflow._impl.nodes.image.filters import ImageBlur
from comfy_ui_api_workflow._impl.workflow import Workflow
from pipelines import ImagesToVideo, TextToImage


def build() -> Workflow:
    workflow = Workflow()

    image = workflow.add_pipeline(
        TextToImage, prompt="a koi pond seen from above, clear water"
    ).output["IMAGE"]

    workflow.add_node(SaveImage, images=image, filename_prefix="koi")
    blurred = workflow.add_node(ImageBlur, image=image, blur_radius=4, sigma=2.0)
    frames = workflow.add_node(
        RepeatImageBatch, image=blurred.output["IMAGE"], amount=16
    )

    workflow.add_pipeline(
        ImagesToVideo, frames=frames.output["IMAGE"], fps=8.0, prefix="video/koi"
    )
    return workflow


if __name__ == "__main__":
    print(json.dumps(build().get_comfy_ui_workflow_format(), indent=2))
