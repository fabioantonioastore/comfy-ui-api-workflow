from typing import Any
from abc import ABC, abstractmethod

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


class BaseNode(ABC):
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id

    @abstractmethod
    def get_comfy_ui_workflow_format(self) -> dict[str, Any]:
        pass
