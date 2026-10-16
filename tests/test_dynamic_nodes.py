import json
from string import ascii_lowercase

import pytest

from comfy_ui_api_workflow._impl.errors.node import (
    InvalidLinkError,
    InvalidNodeParameterValue,
)
from comfy_ui_api_workflow._impl.nodes.model.loaders import CheckpointLoaderSimple
from comfy_ui_api_workflow._impl.nodes.utilities import ComfyMathExpression
from comfy_ui_api_workflow._impl.nodes.utilities.primitive import PrimitiveInt
from comfy_ui_api_workflow._impl.nodes.video import LoadVideo, SaveVideo
from comfy_ui_api_workflow._impl.workflow import Workflow

VIDEO = ("CreateVideo_0", 0)
FPS = ("PrimitiveFloat_1", 0)
SECONDS = ("PrimitiveFloat_2", 0)


def sent(node):
    return node.get_comfy_ui_workflow_format()[node.node_id]


def test_math_expression_sends_each_value_as_values_dot_name():
    node = ComfyMathExpression(
        node_id="129:163",
        expression="floor (a * b + 1)",
        values_a=FPS,
        values_b=SECONDS,
    )

    assert sent(node) == {
        "class_type": "ComfyMathExpression",
        "inputs": {
            "expression": "floor (a * b + 1)",
            "values.a": FPS,
            "values.b": SECONDS,
        },
    }


def test_math_expression_defaults_to_a_plus_b_and_needs_only_values_a():
    node = ComfyMathExpression(node_id="1", values_a=FPS)

    assert sent(node)["inputs"] == {"expression": "a + b", "values.a": FPS}


def test_math_expression_takes_a_value_for_each_letter():
    node = ComfyMathExpression(node_id="1", values_a=FPS, values_z=SECONDS)

    assert sent(node)["inputs"]["values.z"] == SECONDS
    assert ComfyMathExpression.INPUT_NAMES == {
        f"values_{letter}": f"values.{letter}" for letter in ascii_lowercase
    }


def test_math_expression_values_must_be_links():
    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        ComfyMathExpression(node_id="1", values_a=FPS, values_b=2.5)

    assert str(excinfo.value) == "values_b must be a link or None, not float"


def test_math_expression_outputs_a_float_an_int_and_a_bool():
    node = ComfyMathExpression(node_id="1", values_a=FPS)

    assert [node.output[name] for name in ("FLOAT", "INT", "BOOL")] == [
        ("1", 0),
        ("1", 1),
        ("1", 2),
    ]
    assert ComfyMathExpression.RETURN_TYPES == ("FLOAT", "INT", "BOOLEAN")


def test_math_expression_takes_numbers_but_not_other_sockets():
    workflow = Workflow()
    number = workflow.add_node(PrimitiveInt, value=16)
    math = workflow.add_node(ComfyMathExpression, values_a=number.output["INT"])
    model = workflow.add_node(CheckpointLoaderSimple, ckpt_name="model.safetensors")

    with pytest.raises(InvalidLinkError) as excinfo:
        workflow.add_node(ComfyMathExpression, values_a=model.output["MODEL"])

    assert workflow.dependencies(math.node_id) == {number.node_id}
    assert str(excinfo.value) == (
        "Cannot add node 'ComfyMathExpression_3': input 'values_a' uses output 0 of "
        "node 'CheckpointLoaderSimple_2', which is MODEL, but the input takes "
        "FLOAT,INT,BOOLEAN."
    )


def test_save_video_defaults_match_comfy_ui():
    node = SaveVideo(node_id="108", video=VIDEO)

    assert sent(node) == {
        "class_type": "SaveVideo",
        "inputs": {
            "video": VIDEO,
            "filename_prefix": "video/ComfyUI",
            "format": "auto",
            "format.codec": "auto",
        },
    }


def test_save_video_sends_the_chosen_branch_with_dotted_names():
    node = SaveVideo(
        node_id="108",
        video=VIDEO,
        format="mp4",
        format_codec="h264",
        format_codec_encoding="re-encode",
        format_codec_encoding_crf=18.0,
    )

    assert sent(node)["inputs"] == {
        "video": VIDEO,
        "filename_prefix": "video/ComfyUI",
        "format": "mp4",
        "format.codec": "h264",
        "format.codec.encoding": "re-encode",
        "format.codec.encoding.crf": 18.0,
    }


def test_save_video_sends_the_flat_codec_group_with_its_names():
    node = SaveVideo(
        node_id="108",
        video=VIDEO,
        codec="av1",
        codec_encoding="re-encode",
        codec_encoding_crf=60.0,
    )

    assert sent(node)["inputs"] == {
        "video": VIDEO,
        "filename_prefix": "video/ComfyUI",
        "format": "auto",
        "format.codec": "auto",
        "codec": "av1",
        "codec.encoding": "re-encode",
        "codec.encoding.crf": 60.0,
    }


def test_save_video_crf_follows_the_range_of_the_codec():
    av1 = SaveVideo(
        node_id="108",
        video=VIDEO,
        format_codec="av1",
        format_codec_encoding="re-encode",
        format_codec_encoding_crf=63.0,
    )
    linked = SaveVideo(
        node_id="109",
        video=VIDEO,
        format_codec="h264",
        format_codec_encoding="re-encode",
        format_codec_encoding_crf=("PrimitiveFloat_3", 0),
    )

    assert av1.format_codec_encoding_crf == 63.0
    assert linked.format_codec_encoding_crf == ("PrimitiveFloat_3", 0)


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        pytest.param(
            {"format": "avi"},
            "Invalid format, must be one of the following values: "
            "['auto', 'mkv', 'mp4', 'webm']",
            id="unknown-format",
        ),
        pytest.param(
            {"format": "webm", "format_codec": "h264"},
            "format_codec 'h264' is not available for format 'webm'; "
            "use one of ['auto', 'av1']",
            id="codec-of-another-format",
        ),
        pytest.param(
            {"format_codec": "vp9"},
            "format_codec 'vp9' is not available for format 'auto'; "
            "use one of ['auto', 'av1', 'h264']",
            id="unknown-codec",
        ),
        pytest.param(
            {"format_codec_encoding": "re-encode"},
            "format_codec_encoding only applies when format_codec is h264 or av1, "
            "not 'auto'",
            id="encoding-without-a-codec-that-takes-it",
        ),
        pytest.param(
            {"format_codec": "h264", "format_codec_encoding": "fast"},
            "Invalid format_codec_encoding, must be one of the following values: "
            "['auto', 're-encode']",
            id="unknown-encoding",
        ),
        pytest.param(
            {"format_codec": "h264", "format_codec_encoding": "re-encode"},
            "format_codec_encoding_crf is required when format_codec_encoding is "
            "'re-encode'",
            id="re-encode-without-crf",
        ),
        pytest.param(
            {
                "format_codec": "h264",
                "format_codec_encoding": "auto",
                "format_codec_encoding_crf": 18.0,
            },
            "format_codec_encoding_crf only applies when format_codec_encoding is "
            "'re-encode'",
            id="crf-without-re-encode",
        ),
        pytest.param(
            {
                "format_codec": "h264",
                "format_codec_encoding": "re-encode",
                "format_codec_encoding_crf": 52.0,
            },
            "format_codec_encoding_crf must be in closed range: [0.0, 51.0]",
            id="crf-above-the-h264-maximum",
        ),
        pytest.param(
            {
                "format_codec": "av1",
                "format_codec_encoding": "re-encode",
                "format_codec_encoding_crf": 64.0,
            },
            "format_codec_encoding_crf must be in closed range: [0.0, 63.0]",
            id="crf-above-the-av1-maximum",
        ),
        pytest.param(
            {"codec_encoding": "auto"},
            "codec_encoding only applies when codec is h264 or av1, not None",
            id="flat-encoding-without-codec",
        ),
        pytest.param(
            {"codec": "h264", "codec_encoding": "re-encode"},
            "codec_encoding_crf is required when codec_encoding is 're-encode'",
            id="flat-re-encode-without-crf",
        ),
        pytest.param(
            {"codec": "vp9"},
            "Invalid codec, must be one of the following values: "
            "['auto', 'av1', 'h264']",
            id="unknown-flat-codec",
        ),
        pytest.param(
            {"format": ("PrimitiveString_0", 0)},
            "format must be a str, not tuple",
            id="link-for-a-branch-choice",
        ),
    ],
)
def test_save_video_rejects_inputs_outside_the_chosen_branch(inputs, message):
    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        SaveVideo(node_id="108", video=VIDEO, **inputs)

    assert str(excinfo.value) == message


def test_save_video_codec_and_crf_change_together_with_update_node():
    workflow = Workflow()
    clip = workflow.add_node(LoadVideo, file="clip.mp4")
    node = workflow.add_node(
        SaveVideo,
        video=clip.output["VIDEO"],
        format_codec="av1",
        format_codec_encoding="re-encode",
        format_codec_encoding_crf=60.0,
    )

    with pytest.raises(InvalidNodeParameterValue):
        node.format_codec = "h264"
    workflow.update_node(
        node.node_id, format_codec="h264", format_codec_encoding_crf=40.0
    )

    assert (node.format_codec, node.format_codec_encoding_crf) == ("h264", 40.0)


def test_dynamic_nodes_are_loaded_from_the_comfy_ui_format():
    workflow_format = {
        "1": {"class_type": "PrimitiveFloat", "inputs": {"value": 16}},
        "2": {"class_type": "PrimitiveFloat", "inputs": {"value": 5}},
        "3": {
            "class_type": "ComfyMathExpression",
            "inputs": {
                "expression": "floor (a * b + 1)",
                "values.a": ["1", 0],
                "values.b": ["2", 0],
            },
        },
        "4": {"class_type": "LoadVideo", "inputs": {"file": "clip.mp4"}},
        "5": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["4", 0],
                "filename_prefix": "video/Wan2.2_i2v",
                "format": "auto",
                "format.codec": "auto",
                "codec": "auto",
            },
        },
    }
    workflow = Workflow()

    workflow.load_workflow(workflow_format)

    assert workflow.dependencies("3") == {"1", "2"}
    assert workflow.dependencies("5") == {"4"}
    assert json.loads(json.dumps(workflow.get_comfy_ui_workflow_format())) == (
        workflow_format
    )


def test_math_expression_takes_every_letter():
    links = {
        f"values_{letter}": (f"PrimitiveFloat_{letter}", 0)
        for letter in ascii_lowercase
    }

    node = ComfyMathExpression(node_id="1", **links)

    assert sent(node)["inputs"] == {
        "expression": "a + b",
        **{
            f"values.{letter}": (f"PrimitiveFloat_{letter}", 0)
            for letter in ascii_lowercase
        },
    }


def re_encoded_video(workflow):
    clip = workflow.add_node(LoadVideo, file="clip.mp4")
    return workflow.add_node(
        SaveVideo,
        video=clip.output["VIDEO"],
        format_codec="av1",
        format_codec_encoding="re-encode",
        format_codec_encoding_crf=60.0,
    )


def test_crf_cannot_be_removed_while_the_encoding_is_re_encode():
    node = re_encoded_video(Workflow())

    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        del node.format_codec_encoding_crf

    assert str(excinfo.value) == (
        "format_codec_encoding_crf is required when format_codec_encoding is "
        "'re-encode'"
    )
    assert node.format_codec_encoding_crf == 60.0


def test_rejected_update_of_save_video_restores_every_input():
    workflow = Workflow()
    node = re_encoded_video(workflow)
    before = node.inputs()

    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        workflow.update_node(
            node.node_id, format_codec="h264", format_codec_encoding_crf=55.0
        )

    assert str(excinfo.value) == (
        "format_codec_encoding_crf must be in closed range: [0.0, 51.0]"
    )
    assert node.inputs() == before
