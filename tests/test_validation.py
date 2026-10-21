from typing import Any, Literal, Union

import pytest

from comfy_ui_api_workflow._impl.errors.node import InvalidNodeParameterValue
from comfy_ui_api_workflow._impl.nodes.validation import (
    is_link,
    validate_bool,
    validate_choice,
    validate_not_empty,
    validate_range,
    validate_type,
)

LINK = ("PrimitiveInt_0", 0)


@pytest.mark.parametrize("value", [("Loader_0", 0), ("129:94", 2), ["Loader_0", 1]])
def test_node_id_and_output_index_pairs_are_links(value):
    assert is_link(value)


@pytest.mark.parametrize(
    "value",
    [0, "Loader_0", ("Loader_0",), (0, 1), ("Loader_0", "0"), [], [{"x": 0}]],
)
def test_other_values_are_not_links(value):
    assert not is_link(value)


@pytest.mark.parametrize(
    ("value", "range_min", "range_max"),
    [
        (0, 0, 10),
        (10, 0, 10),
        (0.5, 0.0, 1.0),
        (-5, None, 10),
        (10**30, 0, None),
        (7, None, None),
    ],
)
def test_validate_range_accepts_values_in_the_closed_range(value, range_min, range_max):
    validate_range("steps", value, range_min, range_max)


@pytest.mark.parametrize(
    ("value", "range_min", "range_max", "message"),
    [
        (-1, 0, 10, r"steps must be in closed range: \[0, 10\]"),
        (11, 0, 10, r"steps must be in closed range: \[0, 10\]"),
        (11, None, 10, r"steps must be in closed range: \[-inf, 10\]"),
        (-1, 0, None, r"steps must be in closed range: \[0, inf\]"),
    ],
)
def test_validate_range_rejects_values_outside_the_range(
    value, range_min, range_max, message
):
    with pytest.raises(InvalidNodeParameterValue, match=message):
        validate_range("steps", value, range_min, range_max)


@pytest.mark.parametrize("value", [True, False])
def test_validate_range_rejects_bool(value):
    with pytest.raises(InvalidNodeParameterValue, match="steps must be a number"):
        validate_range("steps", value, 0, 10)


def test_validate_range_rejects_bool_without_bounds():
    with pytest.raises(InvalidNodeParameterValue, match="index must be a number"):
        validate_range("index", True)


@pytest.mark.parametrize(("range_min", "range_max"), [(0.0, 100.0), (None, None)])
def test_validate_range_rejects_nan(range_min, range_max):
    with pytest.raises(InvalidNodeParameterValue, match="cfg must not be NaN"):
        validate_range("cfg", float("nan"), range_min, range_max)


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
@pytest.mark.parametrize(
    ("range_min", "range_max"), [(0.0, None), (None, 1.0), (None, None)]
)
def test_validate_range_rejects_infinity_even_without_that_bound(
    value, range_min, range_max
):
    with pytest.raises(InvalidNodeParameterValue, match="threshold must be finite"):
        validate_range("threshold", value, range_min, range_max)


def test_validate_range_skips_links():
    validate_range("steps", LINK, 1, 10)


def test_validate_choice_accepts_an_option():
    validate_choice("crop", "center", frozenset({"disabled", "center"}))


def test_validate_choice_accepts_non_string_options():
    validate_choice("bit_depth", 8, frozenset({"auto", 8, 10}))


def test_validate_choice_rejects_unknown_option_listing_the_options():
    with pytest.raises(
        InvalidNodeParameterValue,
        match=r"Invalid crop, must be one of the following values: "
        r"\['center', 'disabled'\]",
    ):
        validate_choice("crop", "middle", frozenset({"disabled", "center"}))


def test_validate_choice_skips_links():
    validate_choice("crop", LINK, frozenset({"center"}))


def test_validate_not_empty_accepts_a_file_name():
    validate_not_empty("lora_name", "detail.safetensors")


def test_validate_not_empty_rejects_empty_string():
    with pytest.raises(InvalidNodeParameterValue, match="lora_name must not be empty"):
        validate_not_empty("lora_name", "")


def test_validate_not_empty_skips_links():
    validate_not_empty("lora_name", LINK)


@pytest.mark.parametrize("value", [True, False])
def test_validate_bool_accepts_bools(value):
    validate_bool("enabled", value)


@pytest.mark.parametrize("value", [1, 0, "false", None])
def test_validate_bool_rejects_other_values(value):
    with pytest.raises(InvalidNodeParameterValue, match="enabled must be a bool"):
        validate_bool("enabled", value)


def test_validate_bool_skips_links():
    validate_bool("enabled", LINK)


@pytest.mark.parametrize("value", ["20", None, [1, 2], {"steps": 20}])
def test_validate_range_rejects_values_that_are_not_numbers(value):
    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        validate_range("steps", value, 0, 10)

    assert str(excinfo.value) == f"steps must be a number, not {type(value).__name__}"


@pytest.mark.parametrize("value", [["center"], {"crop": "center"}])
def test_validate_choice_rejects_unhashable_values_listing_the_options(value):
    with pytest.raises(
        InvalidNodeParameterValue,
        match=r"Invalid crop, must be one of the following values: "
        r"\['center', 'disabled'\]",
    ):
        validate_choice("crop", value, frozenset({"disabled", "center"}))


@pytest.mark.parametrize(
    ("annotation", "value"),
    [
        pytest.param(int | tuple[str, int], 20, id="int"),
        pytest.param(int | tuple[str, int], LINK, id="int-link"),
        pytest.param(int | tuple[str, int], ["PrimitiveInt_0", 0], id="int-link-list"),
        pytest.param(float | tuple[str, int], 8, id="float-from-int"),
        pytest.param(float | tuple[str, int], 0.5, id="float"),
        pytest.param(str | tuple[str, int], "model.safetensors", id="str"),
        pytest.param(bool | tuple[str, int], False, id="bool"),
        pytest.param(Literal["auto", 8, 10] | tuple[str, int], 8, id="literal-int"),
        pytest.param(Literal["euler"] | tuple[str, int], "heun", id="literal-type"),
        pytest.param(tuple[str, int], LINK, id="link"),
        pytest.param(tuple[str, int] | None, None, id="optional-link"),
        pytest.param(Union[int, None], None, id="typing-union"),
        pytest.param(Any, object(), id="any"),
        pytest.param(list[int], "not checked", id="unknown-annotation"),
    ],
)
def test_validate_type_accepts_values_of_the_annotated_type(annotation, value):
    validate_type("steps", value, annotation)


@pytest.mark.parametrize(
    ("annotation", "value", "message"),
    [
        pytest.param(
            int | tuple[str, int],
            "20",
            "steps must be an int or a link, not str",
            id="int-str",
        ),
        pytest.param(
            int | tuple[str, int],
            20.0,
            "steps must be an int or a link, not float",
            id="int-float",
        ),
        pytest.param(
            int | tuple[str, int],
            True,
            "steps must be an int or a link, not bool",
            id="int-bool",
        ),
        pytest.param(
            float | tuple[str, int],
            None,
            "steps must be a number or a link, not NoneType",
            id="float-none",
        ),
        pytest.param(
            float | tuple[str, int],
            False,
            "steps must be a number or a link, not bool",
            id="float-bool",
        ),
        pytest.param(
            str | tuple[str, int],
            5,
            "steps must be a str or a link, not int",
            id="str-int",
        ),
        pytest.param(
            bool | tuple[str, int],
            1,
            "steps must be a bool or a link, not int",
            id="bool-int",
        ),
        pytest.param(
            Literal["euler", "heun"] | tuple[str, int],
            ["euler"],
            "steps must be a str or a link, not list",
            id="literal-list",
        ),
        pytest.param(
            Literal["auto", 8, 10] | tuple[str, int],
            8.5,
            "steps must be a str or an int or a link, not float",
            id="literal-mixed",
        ),
        pytest.param(
            tuple[str, int],
            "model.safetensors",
            "steps must be a link, not str",
            id="link-str",
        ),
        pytest.param(
            tuple[str, int],
            ("Loader_0",),
            "steps must be a link, not tuple",
            id="link-short",
        ),
        pytest.param(
            tuple[str, int] | None,
            0,
            "steps must be a link or None, not int",
            id="optional-link-int",
        ),
        pytest.param(
            Union[int, None],
            "20",
            "steps must be an int or None, not str",
            id="typing-union",
        ),
    ],
)
def test_validate_type_rejects_values_of_another_type(annotation, value, message):
    with pytest.raises(InvalidNodeParameterValue) as excinfo:
        validate_type("steps", value, annotation)

    assert str(excinfo.value) == message
