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


def test_get_returns_comfy_ui_link():
    output = NodeOutput(
        node_id="Loader0",
        outputs={OutputTypes.MODEL: 0, OutputTypes.CONDITIONING: 1},
    )

    assert output.get(OutputTypes.MODEL) == ["Loader0", 0]
    assert output.get(OutputTypes.CONDITIONING) == ["Loader0", 1]


def test_get_missing_output_raises_output_not_found():
    output = NodeOutput(node_id="Loader0", outputs={OutputTypes.MODEL: 0})

    with pytest.raises(OutputNotFoundError):
        output.get(OutputTypes.LATENT)


def test_output_not_found_is_library_exception():
    assert issubclass(OutputNotFoundError, OutputException)
    assert issubclass(OutputException, ComfyUIAPIWorkflowException)
