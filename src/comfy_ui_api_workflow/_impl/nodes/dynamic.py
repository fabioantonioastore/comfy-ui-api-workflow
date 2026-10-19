from collections.abc import Callable, Iterator
from typing import Any, ClassVar

from ..errors.node import InvalidNodeParameterValue
from .base import _api_value
from .validation import (
    validate_bool,
    validate_choice,
    validate_not_empty,
    validate_range,
)

_NO_DEFAULT = object()


# An input that only exists inside a branch of a dynamic combo or a row of a
# dynamic group. One with a default is sent with it when not given, as the
# ComfyUI frontend does; ComfyUI rejects a prompt missing a required input.
class Value:
    __slots__ = ("default", "required", "check")

    def __init__(
        self,
        default: Any = _NO_DEFAULT,
        required: bool = True,
        check: Callable[[str, Any], None] | None = None,
    ) -> None:
        self.default = default
        self.required = required
        self.check = check

    def validate(self, name: str, value: Any) -> None:
        if self.check is not None:
            self.check(name, value)


def Number(
    default: Any = _NO_DEFAULT,
    low: int | float | None = None,
    high: int | float | None = None,
    required: bool = True,
) -> Value:
    def check(name: str, value: Any) -> None:
        validate_range(name, value, low, high)

    return Value(default, required, check)


def Choice(
    options: tuple[Any, ...], default: Any = _NO_DEFAULT, required: bool = True
) -> Value:
    choices = frozenset(options)

    def check(name: str, value: Any) -> None:
        validate_choice(name, value, choices)

    return Value(default, required, check)


def ServerChoice(required: bool = True) -> Value:
    return Value(required=required, check=validate_not_empty)


def Flag(default: Any = _NO_DEFAULT, required: bool = True) -> Value:
    return Value(default, required, validate_bool)


def Text(default: Any = _NO_DEFAULT, required: bool = True) -> Value:
    return Value(default, required)


def Link(required: bool = True) -> Value:
    return Value(required=required)


def _one_of(keys: list[str]) -> str:
    return repr(keys[0]) if len(keys) == 1 else f"one of {keys}"


# A dynamic combo: each option opens its own inputs, sent as "<combo>.<input>"
# and set here as <combo>_<input>. An attribute can be in several branches, each
# with its own default and checks; the first option is the default one.
class Combo:
    __slots__ = ("branches",)

    def __init__(self, branches: dict[str, dict[str, "Value | Combo"]]) -> None:
        self.branches = branches

    @property
    def default(self) -> str:
        return next(iter(self.branches))

    def attributes(self, branch: str | None = None) -> set[str]:
        keys = self.branches if branch is None else (branch,)
        found: set[str] = set()
        for key in keys:
            for name, spec in self.branches[key].items():
                found.add(name)
                if isinstance(spec, Combo):
                    found |= spec.attributes()
        return found

    def validate(self, node: Any, name: str, value: Any) -> None:
        validate_choice(name, value, frozenset(self.branches))
        for attribute in sorted(self.attributes() - self.attributes(value)):
            if getattr(node, attribute, None) is not None:
                keys = [
                    key for key in self.branches if attribute in self.attributes(key)
                ]
                raise InvalidNodeParameterValue(
                    f"{attribute} only applies when {name} is {_one_of(keys)}"
                )
        for attribute, spec in self.branches[value].items():
            given = getattr(node, attribute, None)
            if isinstance(spec, Combo):
                chosen = spec.default if given is None else given
                spec.validate(node, attribute, chosen)
            elif given is not None:
                spec.validate(attribute, given)
            elif spec.required and spec.default is _NO_DEFAULT:
                raise InvalidNodeParameterValue(
                    f"{attribute} is required when {name} is {value!r}"
                )

    def defaults(self, node: Any, value: Any) -> Iterator[tuple[str, Any]]:
        for attribute, spec in self.branches[value].items():
            given = getattr(node, attribute, None)
            if isinstance(spec, Combo):
                if given is None:
                    given = spec.default
                    yield attribute, given
                yield from spec.defaults(node, given)
            elif given is None and spec.default is not _NO_DEFAULT:
                yield attribute, spec.default


# A dynamic group: rows of the same fields, sent as "<group>.<row>.<field>" and
# set here as <group>_<row>_<field>. A row is used when any of its fields is set;
# the rows a node needs at least have their required fields as constructor
# parameters, so they are always there.
class Group:
    __slots__ = ("rows", "fields")

    def __init__(self, rows: int, fields: dict[str, Value]) -> None:
        self.rows = rows
        self.fields = fields

    def attribute(self, name: str, row: int, field: str) -> str:
        return f"{name}_{row}_{field}"

    def used_rows(self, node: Any, name: str) -> list[int]:
        return [
            row
            for row in range(self.rows)
            if any(
                getattr(node, self.attribute(name, row, field), None) is not None
                for field in self.fields
            )
        ]

    def validate(self, node: Any, name: str) -> None:
        for row in self.used_rows(node, name):
            for field, spec in self.fields.items():
                attribute = self.attribute(name, row, field)
                given = getattr(node, attribute, None)
                if given is not None:
                    spec.validate(attribute, given)
                elif spec.required and spec.default is _NO_DEFAULT:
                    raise InvalidNodeParameterValue(
                        f"{attribute} is required in row {row} of {name}"
                    )

    def defaults(self, node: Any, name: str) -> Iterator[tuple[str, Any]]:
        for row in self.used_rows(node, name):
            for field, spec in self.fields.items():
                attribute = self.attribute(name, row, field)
                if (
                    getattr(node, attribute, None) is None
                    and spec.default is not _NO_DEFAULT
                ):
                    yield attribute, spec.default


def validate_dynamic(node: Any) -> None:
    for name, spec in type(node).DYNAMIC_INPUTS.items():
        if isinstance(spec, Group):
            spec.validate(node, name)
        else:
            spec.validate(node, name, getattr(node, name))


# Mixed in before BaseNode by the nodes with dynamic combos or groups: the
# ComfyUI format also carries the defaults of the chosen branches and rows,
# which are not stored, so changing a branch never leaves stale inputs behind.
class DynamicInputs:
    __slots__ = ()
    DYNAMIC_INPUTS: ClassVar[dict[str, Combo | Group]] = {}

    def get_comfy_ui_workflow_format(self) -> dict[str, Any]:
        node_type = type(self)
        values = self.inputs()  # type: ignore
        for name, spec in node_type.DYNAMIC_INPUTS.items():
            if isinstance(spec, Group):
                values.update(spec.defaults(self, name))
            else:
                values.update(spec.defaults(self, getattr(self, name)))
        order = [
            name
            for cls in reversed(node_type.__mro__)
            for name in cls.__dict__.get("__slots__", ())
        ]
        inputs = {
            node_type.INPUT_NAMES.get(name, name): _api_value(values[name])  # type: ignore
            for name in order
            if name in values
        }
        return {
            self.node_id: {  # type: ignore
                "class_type": node_type.CLASS_TYPE,  # type: ignore
                "inputs": inputs,
            }
        }
