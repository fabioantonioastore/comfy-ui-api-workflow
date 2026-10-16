import pytest

from comfy_ui_api_workflow._impl.enums import OutputTypes
from comfy_ui_api_workflow._impl.errors.output import OutputNotFoundError
from comfy_ui_api_workflow._impl.exceptions import (
    ComfyUIAPIWorkflowException,
    OutputException,
)
from comfy_ui_api_workflow._impl.nodes.base import NodeOutput


def test_output_types_are_comfy_ui_type_strings():
    assert OutputTypes.MODEL == "MODEL"
    assert OutputTypes.CONDITIONING == "CONDITIONING"
    assert OutputTypes.LATENT == "LATENT"


def test_output_types_format_as_their_value():
    assert str(OutputTypes.LATENT) == "LATENT"
    assert f"{OutputTypes.LATENT}" == "LATENT"


def test_output_by_name_returns_comfy_ui_link():
    output = NodeOutput(node_id="Loader_0", outputs={"MODEL": 0, "CLIP": 1})

    assert output["MODEL"] == ("Loader_0", 0)
    assert output["CLIP"] == ("Loader_0", 1)


def test_outputs_of_the_same_type_are_told_apart_by_name():
    output = NodeOutput(node_id="ControlNet_0", outputs={"positive": 0, "negative": 1})

    assert output["positive"] == ("ControlNet_0", 0)
    assert output["negative"] == ("ControlNet_0", 1)


def test_missing_output_raises_output_not_found_naming_node_and_output():
    output = NodeOutput(node_id="Loader_0", outputs={"MODEL": 0})

    with pytest.raises(OutputNotFoundError, match=r"Loader_0.*LATENT"):
        output["LATENT"]


def test_output_not_found_is_library_exception():
    assert issubclass(OutputNotFoundError, OutputException)
    assert issubclass(OutputException, ComfyUIAPIWorkflowException)
