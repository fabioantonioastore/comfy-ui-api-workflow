import contextlib
import importlib.util
import io
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "guide.md",
    ROOT / "docs" / "api.md",
    ROOT / "docs" / "custom_nodes.md",
]
GENERATOR = ROOT / "docs" / "generate_nodes.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_nodes", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def python_blocks():
    return [
        pytest.param(code, id=f"{path.name}-{number}")
        for path in DOCS
        for number, code in enumerate(
            re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S),
            start=1,
        )
    ]


def test_node_catalog_matches_the_library():
    catalog = (ROOT / "docs" / "nodes.md").read_text(encoding="utf-8")

    assert (
        catalog == load_generator().render()
    ), "docs/nodes.md is out of date: run `hatch run python docs/generate_nodes.py`"


def test_node_catalog_lists_every_node_once():
    from comfy_ui_api_workflow._impl.workflow import _node_types

    catalog = load_generator().render()
    rows = re.findall(r"^\| `(\w+)`", catalog, re.M)

    assert sorted(rows) == sorted(cls.__name__ for cls in _node_types().values())


@pytest.mark.parametrize("code", python_blocks())
def test_doc_examples_run_and_print_what_their_comments_say(code):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "<doc example>", "exec"), {"__name__": "doc_example"})

    printed = output.getvalue().splitlines()
    expected = re.findall(r"^[ \t]*print\(.*\)[ \t]+# (.+)$", code, re.M)
    for value in expected:
        assert value in printed


def public_members():
    from comfy_ui_api_workflow import Pipeline, Workflow, enums, errors, nodes
    from comfy_ui_api_workflow._impl.pipeline import PipelineOutput

    classes = [
        Workflow,
        Pipeline,
        PipelineOutput,
        nodes.base.BaseNode,
        nodes.base.NodeOutput,
        enums.OutputTypes,
        errors.node.NodeHasDependentsError,
        errors.node.InvalidNodeChangeError,
        errors.node.InvalidLinkError,
        errors.pipeline.PipelineError,
        errors.workflow.InvalidWorkflowError,
        errors.data_structure.CycleError,
    ]
    inherited = set(dir(Exception)) | set(dir(str))
    members = [
        f"{cls.__name__}.{name}"
        for cls in classes
        for name in sorted(vars(cls))
        if not name.startswith("_") and name not in inherited
    ]
    # Set in __init__, so they are not in the class namespace.
    members += ["Workflow.nodes", "Pipeline.nodes"]
    members += [
        f"validation.{name}"
        for name in sorted(vars(nodes.validation))
        if name.startswith(("validate_", "is_link", "accepts_link"))
    ]
    return members


@pytest.mark.parametrize("member", public_members())
def test_api_reference_documents_every_public_member(member):
    reference = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")

    assert f"`{member}" in reference
