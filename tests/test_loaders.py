import pytest

from comfy_ui_api_workflow._impl.errors.node import InvalidNodeParameterValue
from comfy_ui_api_workflow._impl.nodes.model.loaders import CheckpointLoaderSimple


def make_checkpoint(ckpt_name="sd_xl_base_1.0.safetensors"):
    return CheckpointLoaderSimple(
        node_id="CheckpointLoaderSimple_0", ckpt_name=ckpt_name
    )


def test_comfy_ui_workflow_format():
    assert make_checkpoint().get_comfy_ui_workflow_format() == {
        "CheckpointLoaderSimple_0": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
        }
    }


@pytest.mark.parametrize(("name", "index"), [("MODEL", 0), ("CLIP", 1), ("VAE", 2)])
def test_outputs_follow_comfy_ui_order(name, index):
    assert make_checkpoint().output[name] == ("CheckpointLoaderSimple_0", index)


def test_category_matches_comfy_ui():
    assert CheckpointLoaderSimple.CATEGORY == "model/loaders"


def test_rejects_empty_ckpt_name():
    with pytest.raises(InvalidNodeParameterValue, match="ckpt_name"):
        make_checkpoint(ckpt_name="")
