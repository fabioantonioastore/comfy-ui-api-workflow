from typing import Any, ClassVar

from ..errors.output import OutputNotFoundError
from ..enums import OutputTypes


class BaseOutput:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id


class NodeOutput:
    def __init__(self, node_id: str, outputs: dict[str, int]) -> None:
        self.node_id = node_id
        self.outputs = outputs

    def get(self, output: OutputTypes) -> list[str | int]:
        try:
            return [self.node_id, self.outputs[output]]
        except KeyError as error:
            raise OutputNotFoundError(error)


class BaseNode:
    __slots__ = ("node_id",)
    CATEGORY: ClassVar[str]

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id

    def get_comfy_ui_workflow_format(self) -> dict[str, Any]:
        inputs = {}
        for name in type(self).__slots__:
            if not (name == "node_id" or name == "output"):
                inputs[name] = getattr(self, name)

        return {"class_type": self.__class__.__name__, "inputs": inputs}
