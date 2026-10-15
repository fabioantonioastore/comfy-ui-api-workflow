import pytest

from comfy_ui_api_workflow._impl.enums import OutputTypes
from comfy_ui_api_workflow._impl.nodes.base import NodeOutput


@pytest.fixture
def model_output():
    return NodeOutput(node_id="Checkpoint0", outputs={OutputTypes.MODEL: 0})


@pytest.fixture
def positive_output():
    return NodeOutput(node_id="Positive1", outputs={OutputTypes.CONDITIONING: 0})


@pytest.fixture
def negative_output():
    return NodeOutput(node_id="Negative2", outputs={OutputTypes.CONDITIONING: 0})


@pytest.fixture
def latent_output():
    return NodeOutput(node_id="Latent3", outputs={OutputTypes.LATENT: 0})


@pytest.fixture
def ksampler_inputs(model_output, positive_output, negative_output, latent_output):
    return {
        "model": model_output,
        "sampler_name": "euler",
        "scheduler": "normal",
        "positive": positive_output,
        "negative": negative_output,
        "latent_image": latent_output,
    }
