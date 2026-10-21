import ast
import json
import pkgutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import comfy_ui_api_workflow as package
from comfy_ui_api_workflow._impl.pipeline import Pipeline
from comfy_ui_api_workflow._impl.workflow import Workflow


def fresh(code: str) -> subprocess.CompletedProcess:
    # A new interpreter, so no other test has imported the modules beforehand.
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True
    )


def test_package_root_exports_workflow_pipeline_and_the_namespaces():
    assert package.Workflow is Workflow
    assert package.Pipeline is Pipeline
    assert set(package.__all__) == {
        "errors",
        "nodes",
        "exceptions",
        "enums",
        "Workflow",
        "Pipeline",
    }


def test_node_modules_are_reached_from_the_package_root():
    result = fresh("""
        import comfy_ui_api_workflow as c
        print(c.nodes.image.EmptyImage.CLASS_TYPE)
        print(c.nodes.video.SaveVideo.CLASS_TYPE)
        print(c.nodes.model.sampling.KSampler.CLASS_TYPE)
        print(c.nodes.audio.LoadAudio.CLASS_TYPE)
        print(c.nodes.model.conditioning.wan.Wan22ImageToVideoLatent.CLASS_TYPE)
        print(c.nodes.partner.image.krea.Krea2StyleReferenceNode.CLASS_TYPE)
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [
        "EmptyImage",
        "SaveVideo",
        "KSampler",
        "LoadAudio",
        "Wan22ImageToVideoLatent",
        "Krea2StyleReferenceNode",
    ]


def test_error_modules_are_reached_from_the_package_root():
    result = fresh("""
        import comfy_ui_api_workflow as c
        print(c.errors.pipeline.PipelineError.__name__)
        print(c.errors.output.OutputNotFoundError.__name__)
        print(c.errors.data_structure.CycleError.__name__)
        print(c.errors.node.InvalidLinkError.__name__)
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [
        "PipelineError",
        "OutputNotFoundError",
        "CycleError",
        "InvalidLinkError",
    ]


def test_classes_reached_from_the_root_are_the_ones_the_library_uses():
    result = fresh("""
        import comfy_ui_api_workflow as c
        from comfy_ui_api_workflow._impl.nodes.image import EmptyImage
        from comfy_ui_api_workflow._impl.errors.pipeline import PipelineError
        print(c.nodes.image.EmptyImage is EmptyImage, c.errors.pipeline.PipelineError is PipelineError)
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["True", "True"]


def test_public_module_paths_work_in_from_imports():
    result = fresh("""
        from comfy_ui_api_workflow.nodes.image import SaveImage
        from comfy_ui_api_workflow.nodes.model.sampling import KSampler
        from comfy_ui_api_workflow.errors.pipeline import PipelineError
        from comfy_ui_api_workflow.exceptions import NodeException
        from comfy_ui_api_workflow.enums import OutputTypes
        from comfy_ui_api_workflow._impl.nodes.image import SaveImage as ImplSaveImage
        from comfy_ui_api_workflow._impl.nodes.model.sampling import KSampler as ImplKSampler
        from comfy_ui_api_workflow._impl.errors.pipeline import PipelineError as ImplPipelineError
        from comfy_ui_api_workflow._impl.exceptions import NodeException as ImplNodeException
        from comfy_ui_api_workflow._impl.enums import OutputTypes as ImplOutputTypes
        print(
            SaveImage is ImplSaveImage,
            KSampler is ImplKSampler,
            PipelineError is ImplPipelineError,
            NodeException is ImplNodeException,
            OutputTypes is ImplOutputTypes,
        )
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["True"] * 5


def test_a_public_module_path_imports_the_module_the_root_reaches():
    result = fresh("""
        import comfy_ui_api_workflow as c
        import comfy_ui_api_workflow.nodes.model.sampling as sampling
        print(sampling is c.nodes.model.sampling, sampling.__name__)
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [
        "True",
        "comfy_ui_api_workflow._impl.nodes.model.sampling",
    ]


def test_a_public_path_to_no_module_is_a_module_not_found_error():
    result = fresh("""
        try:
            import comfy_ui_api_workflow.nodes.no_such_module
        except ModuleNotFoundError as error:
            print(error.name)
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["comfy_ui_api_workflow.nodes.no_such_module"]


def test_public_module_paths_are_registered_once():
    result = fresh("""
        import sys
        import comfy_ui_api_workflow
        from comfy_ui_api_workflow._impl.lazy import public_modules
        public_modules("comfy_ui_api_workflow", ["nodes"])
        print(sum(type(finder).__name__ == "_PublicModules" for finder in sys.meta_path))
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["1"]


@pytest.mark.parametrize("namespace", ["nodes", "errors"])
def test_a_name_that_is_no_module_is_an_attribute_error(namespace):
    result = fresh(f"""
        import comfy_ui_api_workflow as c
        print(hasattr(c.{namespace}, "no_such_module"))
        try:
            c.{namespace}.no_such_module
        except AttributeError as error:
            print(type(error).__name__, "no_such_module" in str(error))
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["False", "AttributeError", "True"]


def test_a_workflow_is_built_with_the_package_root_alone():
    result = fresh("""
        import json
        from comfy_ui_api_workflow import Pipeline, Workflow, nodes

        class Canvas(Pipeline):
            def build(self, color: int):
                image = self.add_node(nodes.image.EmptyImage, width=8, height=8, color=color)
                return {"IMAGE": image.output["IMAGE"]}

        workflow = Workflow()
        canvas = workflow.add_pipeline(Canvas, color=255)
        workflow.add_node(nodes.image.SaveImage, images=canvas.output["IMAGE"], filename_prefix="api")
        print(json.dumps(workflow.get_comfy_ui_workflow_format(), sort_keys=True))
        """)

    assert result.returncode == 0, result.stderr
    workflow_format = json.loads(result.stdout)
    assert sorted(node["class_type"] for node in workflow_format.values()) == [
        "EmptyImage",
        "SaveImage",
    ]
    save = next(
        node for node in workflow_format.values() if node["class_type"] == "SaveImage"
    )
    assert save["inputs"]["filename_prefix"] == "api"


PACKAGE_ROOT = Path(package.__file__).parent
PACKAGES = sorted(
    init.parent
    for namespace in ("errors", "nodes")
    for init in (PACKAGE_ROOT / "_impl" / namespace).rglob("__init__.py")
)


# Editors complete only the submodules a package imports itself.
@pytest.mark.parametrize(
    "directory",
    PACKAGES,
    ids=lambda directory: ".".join(directory.relative_to(PACKAGE_ROOT).parts),
)
def test_a_package_imports_all_its_submodules(directory):
    tree = ast.parse((directory / "__init__.py").read_text())
    imported = {
        alias.name
        for statement in tree.body
        if isinstance(statement, ast.ImportFrom)
        and statement.level == 1
        and statement.module is None
        for alias in statement.names
    }

    assert imported == {
        module.name for module in pkgutil.iter_modules([str(directory)])
    }
