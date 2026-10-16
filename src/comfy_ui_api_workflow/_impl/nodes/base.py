import functools
import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, ClassVar, NamedTuple

from ..errors.node import InvalidNodeParameterValue
from ..errors.output import OutputNotFoundError
from .validation import is_link, validate_type

_UNSET = object()
_INPUT_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
_STATE = ("_on_change", "_built", "_batch")


class _Signature(NamedTuple):
    annotations: dict[str, Any]
    required: frozenset[str]
    rebuildable: bool


@functools.cache
def _signature(node_type: type) -> _Signature:
    parameters = inspect.signature(node_type).parameters.values()
    named = {
        parameter.name: parameter
        for parameter in parameters
        if parameter.kind in _INPUT_KINDS
    }
    annotations = {
        name: parameter.annotation
        for name, parameter in named.items()
        if parameter.annotation is not parameter.empty
    }
    annotations["node_id"] = str
    classes = node_type.__mro__
    slots = {
        name
        for cls in classes[: classes.index(BaseNode)]
        for name in cls.__dict__.get("__slots__", ())
    }
    takes_any = any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters)
    required = {
        name
        for name, parameter in named.items()
        if parameter.default is parameter.empty
    }
    return _Signature(
        annotations=annotations,
        required=frozenset(required),
        # The node can only be checked again by building it from its inputs.
        rebuildable=(takes_any or slots <= set(named))
        and required - {"node_id"} <= slots,
    )


def _tracks_construction(init: Callable[..., None]) -> Callable[..., None]:
    # Only the outermost __init__ marks the node as built, so a subclass that
    # sets inputs after super().__init__() is not checked half built.
    @functools.wraps(init)
    def wrapper(self: "BaseNode", *args: Any, **kwargs: Any) -> None:
        outermost = not hasattr(self, "_built")
        if outermost:
            object.__setattr__(self, "_built", False)
        init(self, *args, **kwargs)
        if outermost:
            object.__setattr__(self, "_built", True)

    return wrapper


def _api_value(value: Any) -> Any:
    # ComfyUI reads every list as a link, so a list widget value goes wrapped.
    if isinstance(value, list) and not is_link(value):
        return {"__value__": value}
    return value


class NodeOutput:
    __slots__ = ("node_id", "outputs")

    def __init__(self, node_id: str, outputs: dict[str, int]) -> None:
        self.node_id = node_id
        self.outputs = outputs

    def __getitem__(self, name: str) -> tuple[str, int]:
        try:
            return (self.node_id, self.outputs[name])
        except KeyError as error:
            raise OutputNotFoundError(
                f"node: {self.node_id} don't have the output: {name}"
            ) from error


class BaseNode:
    __slots__ = ("node_id", "_on_change", "_built", "_batch")
    CATEGORY: ClassVar[str]
    CLASS_TYPE: ClassVar[str]
    OUTPUTS: ClassVar[dict[str, int]] = {}
    INPUT_NAMES: ClassVar[dict[str, str]] = {}
    # ComfyUI socket types, as in object_info: RETURN_TYPES by output index and
    # INPUT_SOCKET_TYPES by attribute name. A node without them is not type checked.
    # A V1 combo, whose type in object_info is its list of options, is "V1 COMBO":
    # no output has that type, so as in ComfyUI only * and match type outputs fit.
    RETURN_TYPES: ClassVar[tuple[str, ...]] = ()
    INPUT_SOCKET_TYPES: ClassVar[dict[str, str]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Nodes whose ComfyUI name is not an identifier, like "Epsilon Scaling",
        # declare CLASS_TYPE; every other node uses its class name.
        if "CLASS_TYPE" not in cls.__dict__:
            cls.CLASS_TYPE = cls.__name__
        if "__init__" in cls.__dict__:
            cls.__init__ = _tracks_construction(cls.__dict__["__init__"])

    @_tracks_construction
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self._on_change: Callable[[BaseNode], None] | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        # The type check runs before the value is stored, so a wrong type never
        # gets in, whether at construction or in a later change.
        annotation = _signature(type(self)).annotations.get(name)
        if annotation is not None:
            validate_type(name, value, annotation)
        if name in _STATE or not getattr(self, "_built", False):
            super().__setattr__(name, value)
            return
        previous = getattr(self, name, _UNSET)
        super().__setattr__(name, value)
        try:
            self.__changed()
        except Exception:
            # A rejected change is undone, so the node stays as it was.
            if previous is _UNSET:
                super().__delattr__(name)
            else:
                super().__setattr__(name, previous)
            raise

    def __delattr__(self, name: str) -> None:
        if name in _STATE or not getattr(self, "_built", False):
            super().__delattr__(name)
            return
        if name in _signature(type(self)).required:
            raise InvalidNodeParameterValue(
                f"{name} cannot be removed because it is not optional"
            )
        previous = getattr(self, name)
        super().__delattr__(name)
        try:
            self.__changed(removed=name)
        except Exception:
            super().__setattr__(name, previous)
            raise

    def __changed(self, removed: str | None = None) -> None:
        if not getattr(self, "_batch", False):
            rebuilt = self._validate()
            # If the constructor fills a removed input back in, the class always
            # sends it, like a default or a dict that None stands for.
            if (
                removed is not None
                and rebuilt is not None
                and removed in rebuilt.inputs()
            ):
                raise InvalidNodeParameterValue(
                    f"{removed} cannot be removed because it is not optional"
                )
        on_change = getattr(self, "_on_change", None)
        if on_change is not None:
            on_change(self)

    def _validate(self) -> "BaseNode | None":
        # Building a node runs every check of its class, so a copy built from the
        # current inputs tells if they are still valid together.
        if _signature(type(self)).rebuildable:
            return type(self)(node_id=self.node_id, **self.inputs())
        return None

    @contextmanager
    def _changing(self) -> Iterator[None]:
        # Inside the block each change is still type checked and linked, but the
        # node is only validated again by the caller, once, at the end.
        batch = getattr(self, "_batch", False)
        object.__setattr__(self, "_batch", True)
        try:
            yield
        finally:
            object.__setattr__(self, "_batch", batch)

    @property
    def output(self) -> NodeOutput:
        return NodeOutput(node_id=self.node_id, outputs=type(self).OUTPUTS)

    def inputs(self) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        node_classes = type(self).__mro__
        # Every slot declared below BaseNode is a ComfyUI input.
        for cls in reversed(node_classes[: node_classes.index(BaseNode)]):
            for name in cls.__dict__.get("__slots__", ()):
                if hasattr(self, name):
                    inputs[name] = getattr(self, name)
        return inputs

    def get_comfy_ui_workflow_format(self) -> dict[str, Any]:
        node_type = type(self)
        inputs = {
            node_type.INPUT_NAMES.get(name, name): _api_value(value)
            for name, value in self.inputs().items()
        }
        return {self.node_id: {"class_type": node_type.CLASS_TYPE, "inputs": inputs}}
