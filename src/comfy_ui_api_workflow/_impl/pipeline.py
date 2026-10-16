from typing import TYPE_CHECKING, Any, TypeVar

from .errors.node import quote_names
from .errors.output import OutputNotFoundError
from .errors.pipeline import PipelineError
from .nodes.base import BaseNode

if TYPE_CHECKING:
    from .workflow import Workflow

NodeType = TypeVar("NodeType", bound=BaseNode)


class PipelineOutput:
    __slots__ = ("pipeline", "outputs")

    def __init__(self, pipeline: str, outputs: dict[str, tuple[str, int]]) -> None:
        self.pipeline = pipeline
        self.outputs = outputs

    def __getitem__(self, name: str) -> tuple[str, int]:
        if name in self.outputs:
            return self.outputs[name]
        message = f"Pipeline {self.pipeline!r} has no output {name!r}."
        if self.outputs:
            message += f" Its outputs are: {quote_names(list(self.outputs))}."
        raise OutputNotFoundError(message)


class Pipeline:
    def __init__(self) -> None:
        self.nodes: dict[str, BaseNode] = {}
        self.__workflow: "Workflow | None" = None
        self.__outputs: dict[str, tuple[str, int]] = {}
        self.__rejection: tuple[int, type, Exception] | None = None

    @property
    def output(self) -> PipelineOutput:
        return PipelineOutput(type(self).__name__, self.__outputs)

    def build(self, **parameters: Any) -> dict[str, tuple[str, int]] | None:
        raise NotImplementedError

    def add_node(self, node_type: type[NodeType], **kwargs: Any) -> NodeType:
        if self.__workflow is None:
            reason = (
                "add_node only works inside build, "
                "while Workflow.add_pipeline adds the pipeline"
            )
            raise PipelineError(type(self).__name__, reason)
        try:
            node = self.__workflow.add_node(node_type, **kwargs)
        except Exception as error:
            if self.__rejection is None:
                self.__rejection = (len(self.nodes) + 1, node_type, error)
            raise
        self.nodes[node.node_id] = node
        return node

    def _bind(self, workflow: "Workflow | None") -> None:
        self.__workflow = workflow

    def _rejection(self) -> tuple[int, type, Exception] | None:
        return self.__rejection

    def _set_outputs(self, outputs: dict[str, tuple[str, int]]) -> None:
        self.__outputs = outputs
