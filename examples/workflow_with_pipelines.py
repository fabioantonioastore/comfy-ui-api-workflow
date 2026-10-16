import json

from comfy_ui_api_workflow._impl.workflow import Workflow
from pipelines import SaveImages, TextToImage, Upscale


def build() -> Workflow:
    workflow = Workflow()

    image = workflow.add_pipeline(
        TextToImage, prompt="a cozy cabin in a snowy forest, warm light"
    ).output["IMAGE"]
    upscaled = workflow.add_pipeline(Upscale, image=image, scale=1.5).output["IMAGE"]
    workflow.add_pipeline(SaveImages, images=upscaled, prefix="cabin")
    return workflow


if __name__ == "__main__":
    print(json.dumps(build().get_comfy_ui_workflow_format(), indent=2))
