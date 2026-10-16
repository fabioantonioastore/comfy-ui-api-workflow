import json
import subprocess
import sys
from pathlib import Path

import pytest

from comfy_ui_api_workflow._impl.workflow import Workflow

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
TEXT_TO_IMAGE = [
    "CheckpointLoaderSimple",
    "CLIPTextEncode",
    "CLIPTextEncode",
    "EmptyLatentImage",
    "KSampler",
    "VAEDecode",
]


def run_example(name):
    # Run as a script, like a user would, so its own imports are checked too.
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def class_types(workflow_format):
    return sorted(node["class_type"] for node in workflow_format.values())


@pytest.mark.parametrize(
    "name",
    [
        "workflow_with_nodes.py",
        "workflow_with_pipelines.py",
        "workflow_with_pipelines_and_nodes.py",
    ],
)
def test_example_prints_a_workflow_that_loads_back(name):
    workflow_format = json.loads(run_example(name))
    workflow = Workflow()

    workflow.load_workflow(workflow_format)

    assert json.loads(json.dumps(workflow.get_comfy_ui_workflow_format())) == (
        workflow_format
    )


def test_workflow_with_nodes_is_text_to_image_saved_to_a_file():
    workflow_format = json.loads(run_example("workflow_with_nodes.py"))

    assert class_types(workflow_format) == sorted([*TEXT_TO_IMAGE, "SaveImage"])
    assert workflow_format["KSampler_4"]["inputs"]["steps"] == 30


def test_workflow_with_pipelines_chains_three_pipelines():
    workflow_format = json.loads(run_example("workflow_with_pipelines.py"))

    assert class_types(workflow_format) == sorted(
        [*TEXT_TO_IMAGE, "ImageScaleBy", "SaveImage"]
    )
    assert workflow_format["ImageScaleBy_6"]["inputs"]["image"] == ["VAEDecode_5", 0]
    assert workflow_format["SaveImage_7"]["inputs"]["images"] == ["ImageScaleBy_6", 0]


def test_workflow_with_pipelines_and_nodes_links_them_both_ways():
    workflow_format = json.loads(run_example("workflow_with_pipelines_and_nodes.py"))

    assert class_types(workflow_format) == sorted(
        [
            *TEXT_TO_IMAGE,
            "SaveImage",
            "ImageBlur",
            "RepeatImageBatch",
            "CreateVideo",
            "SaveVideo",
        ]
    )
    assert workflow_format["ImageBlur_7"]["inputs"]["image"] == ["VAEDecode_5", 0]
    assert workflow_format["CreateVideo_9"]["inputs"]["images"] == [
        "RepeatImageBatch_8",
        0,
    ]


def test_building_pipelines_example_shows_what_add_pipeline_checks():
    lines = run_example("building_pipelines.py").splitlines()

    assert lines == [
        "nodes: ['CheckpointLoaderSimple_0', 'CLIPTextEncode_1', 'CLIPTextEncode_2', "
        "'EmptyLatentImage_3', 'KSampler_4', 'VAEDecode_5']",
        "IMAGE output: ('VAEDecode_5', 0)",
        "Cannot add pipeline 'TextToImage': it is missing the parameter 'prompt'. "
        "No node of the pipeline was added.",
        "Cannot add pipeline 'Upscale': node 1 of the pipeline (ImageScaleBy) was "
        "rejected: scale_by must be in closed range: [0.01, 8.0]. "
        "No node of the pipeline was added.",
        "nodes in the workflow: 6",
    ]
