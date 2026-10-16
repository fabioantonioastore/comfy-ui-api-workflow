from comfy_ui_api_workflow._impl.errors.pipeline import PipelineError
from comfy_ui_api_workflow._impl.workflow import Workflow
from pipelines import TextToImage, Upscale


def main() -> None:
    workflow = Workflow()
    pipeline = workflow.add_pipeline(TextToImage, prompt="a lighthouse at dusk")
    print("nodes:", list(pipeline.nodes))
    print("IMAGE output:", pipeline.output["IMAGE"])
    try:
        workflow.add_pipeline(TextToImage)
    except PipelineError as error:
        print(error)
    try:
        workflow.add_pipeline(Upscale, image=pipeline.output["IMAGE"], scale=100.0)
    except PipelineError as error:
        print(error)
    print("nodes in the workflow:", len(workflow.nodes))


if __name__ == "__main__":
    main()
