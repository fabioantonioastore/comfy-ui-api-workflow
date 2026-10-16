import math
from types import NoneType, UnionType
from typing import Any, Literal, Union, get_args, get_origin

from ..errors.node import InvalidNodeParameterValue

_TYPE_NAMES = {bool: "a bool", int: "an int", float: "a number", str: "a str"}


def is_link(value: Any) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def _options(annotation: Any) -> tuple[Any, ...]:
    if get_origin(annotation) in (Union, UnionType):
        return get_args(annotation)
    return (annotation,)


def _accepts(option: Any, value: Any) -> bool:
    if get_origin(option) is tuple:
        return is_link(value)
    if get_origin(option) is Literal:
        return any(_accepts(type(choice), value) for choice in get_args(option))
    if option is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if option is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if option in (bool, str, NoneType):
        return isinstance(value, option)
    # Annotations this check does not know, such as Any, accept every value.
    return True


def accepts_link(annotation: Any) -> bool:
    return any(get_origin(option) is tuple for option in _options(annotation))


def _describe(option: Any) -> list[str]:
    if get_origin(option) is tuple:
        return ["a link"]
    if get_origin(option) is Literal:
        return [
            description
            for choice in get_args(option)
            for description in _describe(type(choice))
        ]
    if option is NoneType:
        return ["None"]
    return [_TYPE_NAMES[option]] if option in _TYPE_NAMES else []


# The type comes from the constructor annotation, like int | tuple[str, int].
# A Literal only checks the type of its options: validate_choice checks the value.
def validate_type(name: str, value: Any, annotation: Any) -> None:
    options = _options(annotation)
    if any(_accepts(option, value) for option in options):
        return
    expected = dict.fromkeys(
        description for option in options for description in _describe(option)
    )
    raise InvalidNodeParameterValue(
        f"{name} must be {' or '.join(expected)}, not {type(value).__name__}"
    )


# Every validator lets a link through: its value only exists when ComfyUI runs.
def validate_range(
    name: str,
    value: Any,
    range_min: int | float | None = None,
    range_max: int | float | None = None,
) -> None:
    if is_link(value):
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidNodeParameterValue(
            f"{name} must be a number, not {type(value).__name__}"
        )
    # NaN fails every comparison, so no range would ever stop it.
    if value != value:
        raise InvalidNodeParameterValue(f"{name} must not be NaN")
    # The API format is JSON, which has no infinity, even where ComfyUI sets no bound.
    if value in (math.inf, -math.inf):
        raise InvalidNodeParameterValue(f"{name} must be finite, not {value}")
    if (range_min is not None and value < range_min) or (
        range_max is not None and value > range_max
    ):
        lower = "-inf" if range_min is None else range_min
        upper = "inf" if range_max is None else range_max
        raise InvalidNodeParameterValue(
            f"{name} must be in closed range: [{lower}, {upper}]"
        )


def validate_choice(name: str, value: Any, choices: frozenset[Any]) -> None:
    if is_link(value):
        return
    try:
        valid = value in choices
    except TypeError:
        valid = False
    if not valid:
        options = sorted(choices, key=str)
        raise InvalidNodeParameterValue(
            f"Invalid {name}, must be one of the following values: {options}"
        )


def validate_not_empty(name: str, value: Any) -> None:
    if is_link(value):
        return
    if not value:
        raise InvalidNodeParameterValue(f"{name} must not be empty")


def validate_bool(name: str, value: Any) -> None:
    if is_link(value):
        return
    if not isinstance(value, bool):
        raise InvalidNodeParameterValue(f"{name} must be a bool")
