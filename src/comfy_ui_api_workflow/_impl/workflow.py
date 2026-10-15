from typing import Any

from .nodes.base import BaseNode, NodeOutput
from .errors.node import NodeNotFoundError, NodeNotHaveOutput


class Workflow:
    def __init__(self) -> None:
        self.comfy_ui_workflow: dict[str, Any] = {}
        self.nodes: dict[str, BaseNode] = {}
        self.__next_index = 0

    def get_node_output(self, node_id: str) -> NodeOutput:
        if node_id not in self.nodes:
            raise NodeNotFoundError(f"node: {node_id} not found")
        try:
            return getattr(self.nodes[node_id], "output")
        except AttributeError:
            raise NodeNotHaveOutput(f"node: {node_id} don't have a output")

    def add_node(self, node_type: type[BaseNode], **kwargs: Any) -> str:
        node_id = node_type.__name__ + f"_{self.__next_index}"
        self.__next_index += 1
        self.nodes[node_id] = node_type(node_id=node_id, **kwargs)
        self.comfy_ui_workflow[node_id] = self.nodes[
            node_id
        ].get_comfy_ui_workflow_format()
        return node_id
