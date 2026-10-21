import importlib
import inspect
import pkgutil
from collections.abc import Iterator
from functools import cache, partial
from types import MemberDescriptorType
from typing import Any, TypeVar

from . import nodes as nodes_package
from .data_structure import (
    IncrementalDirectedAcyclicGraph,
    IncrementalDirectedAcyclicGraphNode,
)
from .errors.data_structure import CycleError
from .errors.node import (
    InvalidLinkError,
    InvalidNodeChangeError,
    InvalidNodeParameterValue,
    NodeHasDependentsError,
    NodeNotFoundError,
    describe_dependents,
    quote_names,
)
from .errors.pipeline import PipelineError
from .errors.workflow import InvalidWorkflowError
from .nodes.base import BaseNode, NodeOutput
from .nodes.validation import accepts_link, validate_type
from .pipeline import Pipeline

NodeType = TypeVar("NodeType", bound=BaseNode)
PipelineType = TypeVar("PipelineType", bound=Pipeline)
_UNSET = object()
_ANY_SOCKETS = frozenset({"*", "COMFY_MATCHTYPE_V3"})
_NAMED_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def _is_link(value: Any) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2  # type: ignore
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def _is_input(node: BaseNode, name: str) -> bool:
    return name not in BaseNode.__slots__ and isinstance(
        getattr(type(node), name, None), MemberDescriptorType
    )


def _subclasses(node_type: type[BaseNode]) -> Iterator[type[BaseNode]]:
    for subclass in node_type.__subclasses__():
        yield subclass
        yield from _subclasses(subclass)


@cache
def _node_types() -> dict[str, type[BaseNode]]:
    prefix = f"{nodes_package.__name__}."
    for module in pkgutil.walk_packages(nodes_package.__path__, prefix):
        importlib.import_module(module.name)
    return {
        node_type.CLASS_TYPE: node_type
        for node_type in _subclasses(BaseNode)
        if node_type.__module__.startswith(prefix)
    }


def _field(node_id: str, data: dict[str, Any], key: str, kind: type) -> Any:
    if key not in data:
        raise InvalidWorkflowError(node_id, f"{key!r} is missing")
    if not isinstance(data[key], kind):
        reason = f"{key!r} must be a {kind.__name__}, got {type(data[key]).__name__}"
        raise InvalidWorkflowError(node_id, reason)
    return data[key]


def _load_value(node_id: str, name: str, value: Any) -> Any:
    if isinstance(value, dict) and "__value__" in value:
        return value["__value__"]  # type: ignore
    if isinstance(value, (list, tuple)):
        if not _is_link(value):
            reason = f"input {name!r} is a list but not a link [node_id, output_index]"
            raise InvalidWorkflowError(node_id, reason)
        return tuple(value)  # type: ignore
    return value  # type: ignore


def _load_node(node_id: Any, data: Any) -> BaseNode:
    if not isinstance(node_id, str):
        raise InvalidWorkflowError(None, f"node ids must be str, got {node_id!r}")
    if not isinstance(data, dict):
        reason = (
            "expected a dict with 'class_type' and 'inputs', "
            f"got {type(data).__name__}"
        )
        raise InvalidWorkflowError(node_id, reason)
    class_type = _field(node_id, data, "class_type", str)  # type: ignore
    inputs = _field(node_id, data, "inputs", dict)  # type: ignore
    node_type = _node_types().get(class_type)
    if node_type is None:
        raise InvalidWorkflowError(node_id, f"unknown class_type {class_type!r}")
    parameters = inspect.signature(node_type).parameters
    names = {
        node_type.INPUT_NAMES.get(parameter, parameter): parameter
        for parameter in parameters
        if parameter != "node_id"
    }
    kwargs: dict[str, Any] = {}
    for name, value in inputs.items():
        if name not in names:
            raise InvalidWorkflowError(node_id, f"{class_type} has no input {name!r}")
        kwargs[names[name]] = _load_value(node_id, name, value)
    missing = [
        name
        for name, parameter in names.items()
        if parameter not in kwargs
        and parameters[parameter].default is inspect.Parameter.empty
    ]
    if missing:
        noun = "input" if len(missing) == 1 else "inputs"
        reason = f"{class_type} is missing the {noun} {quote_names(missing)}"
        raise InvalidWorkflowError(node_id, reason)
    try:
        return node_type(node_id=node_id, **kwargs)
    except InvalidNodeParameterValue as error:
        raise InvalidWorkflowError(node_id, str(error)) from error


def _sockets_match(output_type: str, input_type: str) -> bool:
    outputs = {part.strip() for part in output_type.split(",")}
    inputs = {part.strip() for part in input_type.split(",")}
    return bool((outputs | inputs) & _ANY_SOCKETS or outputs & inputs)


def _missing_output(source: BaseNode, index: int) -> str | None:
    source_type = type(source)
    if isinstance(index, bool) or index not in source_type.OUTPUTS.values():
        return (
            f"output {index} of node {source.node_id!r}, "
            f"but {source_type.CLASS_TYPE} has no output {index}"
        )
    return None


def _link_problem(
    source: BaseNode, index: int, node: BaseNode, name: str
) -> str | None:
    missing = _missing_output(source, index)
    if missing is not None:
        return missing
    source_type = type(source)
    input_type = type(node).INPUT_SOCKET_TYPES.get(name)
    if input_type is None or index >= len(source_type.RETURN_TYPES):
        return None
    output_type = source_type.RETURN_TYPES[index]
    if _sockets_match(output_type, input_type):
        return None
    return (
        f"output {index} of node {source.node_id!r}, which is {output_type}, "
        f"but the input takes {input_type}"
    )


def _links(node: BaseNode) -> Iterator[tuple[str, str, int]]:
    for name, value in node.inputs().items():
        if _is_link(value):
            yield name, value[0], value[1]


def _comfy_name(node: BaseNode, name: str) -> str:
    return type(node).INPUT_NAMES.get(name, name)


def _check_link(
    nodes: dict[str, BaseNode], node: BaseNode, name: str, source_id: str, index: int
) -> None:
    node_id = node.node_id
    shown = _comfy_name(node, name)
    if source_id == node_id:
        reason = f"input {shown!r} uses the node's own output"
        raise InvalidWorkflowError(node_id, reason)
    if source_id not in nodes:
        reason = (
            f"input {shown!r} uses node {source_id!r}, which is not in the workflow"
        )
        raise InvalidWorkflowError(node_id, reason)
    problem = _link_problem(nodes[source_id], index, node, name)
    if problem is not None:
        raise InvalidWorkflowError(node_id, f"input {shown!r} uses {problem}")


def _sentence(error: Exception) -> str:
    return str(error).rstrip(".")


def _new_pipeline(pipeline_type: Any) -> tuple[str, Pipeline]:
    if isinstance(pipeline_type, Pipeline):
        name = type(pipeline_type).__name__
        reason = "add_pipeline takes the pipeline class, not an instance"
        raise PipelineError(name, reason)
    name = getattr(pipeline_type, "__name__", type(pipeline_type).__name__)
    if not (isinstance(pipeline_type, type) and issubclass(pipeline_type, Pipeline)):
        raise PipelineError(name, "add_pipeline takes a Pipeline subclass")
    if pipeline_type.build is Pipeline.build:
        raise PipelineError(name, "it does not define build")
    return name, pipeline_type()


def _check_parameters(
    name: str,
    pipeline: Pipeline,
    parameters: dict[str, Any],
    nodes: dict[str, BaseNode],
) -> None:
    signature = inspect.signature(pipeline.build).parameters.values()
    named = {
        parameter.name: parameter
        for parameter in signature
        if parameter.kind in _NAMED_KINDS
    }
    takes_any = any(parameter.kind is parameter.VAR_KEYWORD for parameter in signature)
    for given in parameters:
        if given not in named and not takes_any:
            takes = (
                f"it takes {quote_names(list(named))}"
                if named
                else "it takes no parameters"
            )
            raise PipelineError(name, f"it has no parameter {given!r} ({takes})")
    missing = [
        parameter.name
        for parameter in named.values()
        if parameter.default is parameter.empty and parameter.name not in parameters
    ]
    if missing:
        noun = "parameter" if len(missing) == 1 else "parameters"
        raise PipelineError(name, f"it is missing the {noun} {quote_names(missing)}")
    for given, value in parameters.items():
        parameter = named.get(given)
        if parameter is None or parameter.annotation is parameter.empty:
            continue
        try:
            validate_type(given, value, parameter.annotation)
        except InvalidNodeParameterValue as error:
            raise PipelineError(name, str(error)) from error
        if not (accepts_link(parameter.annotation) and _is_link(value)):
            continue
        source_id, index = value
        if source_id not in nodes:
            reason = (
                f"parameter {given!r} uses node {source_id!r}, "
                "which is not in the workflow"
            )
            raise PipelineError(name, reason)
        missing = _missing_output(nodes[source_id], index)
        if missing is not None:
            raise PipelineError(name, f"parameter {given!r} uses {missing}")


def _pipeline_error(name: str, pipeline: Pipeline, error: Exception) -> PipelineError:
    rejection = pipeline._rejection()  # type: ignore
    if rejection is not None:
        position, node_type, rejected = rejection
        reason = (
            f"node {position} of the pipeline ({node_type.__name__}) was rejected: "
            f"{_sentence(rejected)}"
        )
        return PipelineError(name, reason)
    if isinstance(error, InvalidNodeChangeError) and error.node_id in pipeline.nodes:
        position = list(pipeline.nodes).index(error.node_id) + 1
        changed = type(pipeline.nodes[error.node_id]).__name__
        reason = (
            f"a change to node {position} of the pipeline ({changed}) was rejected: "
            f"{_sentence(error)}"
        )
        return PipelineError(name, reason)
    reason = f"its build failed with {type(error).__name__}"
    if str(error):
        reason += f": {_sentence(error)}"
    return PipelineError(name, reason)


class Workflow:
    def __init__(self) -> None:
        self.nodes: dict[str, BaseNode] = {}
        self.__next_index = 0
        self.__graph = IncrementalDirectedAcyclicGraph()
        self.__graph_nodes: dict[str, IncrementalDirectedAcyclicGraphNode] = {}

    def get_node_output(self, node_id: str) -> NodeOutput:
        if node_id not in self.nodes:
            raise NodeNotFoundError(f"node: {node_id} not found")
        return self.nodes[node_id].output

    def dependencies(self, node_id: str) -> set[str]:
        graph_node = self.__graph_node(node_id)
        return {
            dependency.data for dependency in self.__graph.dependencies(node=graph_node)
        }

    def dependents(self, node_id: str) -> set[str]:
        graph_node = self.__graph_node(node_id)
        return {
            dependent.data for dependent in self.__graph.dependents(node=graph_node)
        }

    def add_node(self, node_type: type[NodeType], **kwargs: Any) -> NodeType:
        node_id = self.__new_node_id(node_type)
        node = node_type(node_id=node_id, **kwargs)
        graph_node = IncrementalDirectedAcyclicGraphNode(node_id)
        self.__graph.add_node(node=graph_node)
        self.__graph_nodes[node_id] = graph_node
        try:
            self.__check_links(node)
            self.__link(graph_node=graph_node, node=node)
        except (NodeNotFoundError, CycleError, InvalidLinkError):
            self.__graph.remove_node(node=graph_node)
            del self.__graph_nodes[node_id]
            raise
        self.nodes[node_id] = node
        node._on_change = partial(self.__check_change, graph_node)  # type: ignore
        return node

    def add_pipeline(
        self, pipeline_type: type[PipelineType], **parameters: Any
    ) -> PipelineType:
        name, pipeline = _new_pipeline(pipeline_type)
        _check_parameters(name, pipeline, parameters, self.nodes)
        next_index = self.__next_index
        pipeline._bind(self)  # type: ignore
        try:
            outputs = pipeline.build(**parameters)
            rejection = pipeline._rejection()  # type: ignore
            if rejection is not None:
                raise rejection[2]
            pipeline._set_outputs(self.__pipeline_outputs(name, outputs))  # type: ignore
        except Exception as error:
            failure = error
            if not isinstance(error, PipelineError):
                failure = _pipeline_error(name, pipeline, error)
            for node_id in reversed(list(pipeline.nodes)):
                self.remove_node(node_id)
            pipeline.nodes.clear()
            self.__next_index = next_index
            if failure is error:
                raise
            rejection = pipeline._rejection()  # type: ignore
            raise failure from (error if rejection is None else rejection[2])
        finally:
            pipeline._bind(None)  # type: ignore
        return pipeline  # type: ignore

    def update_node(self, node_id: str, **inputs: Any) -> None:
        if node_id not in self.nodes:
            raise NodeNotFoundError(f"node: {node_id} not found")
        node = self.nodes[node_id]
        changed: list[tuple[str, Any]] = []
        try:
            with node._changing():  # type: ignore
                for name, value in inputs.items():
                    if not _is_input(node, name):
                        reason = f"{type(node).__name__} has no input {name!r}"
                        raise InvalidNodeChangeError(node_id, name, reason)
                    previous = getattr(node, name, _UNSET)
                    setattr(node, name, value)
                    changed.append((name, previous))
            node._validate()  # type: ignore
        except (InvalidNodeChangeError, InvalidNodeParameterValue):
            with node._changing():  # type: ignore
                for name, previous in reversed(changed):
                    if previous is _UNSET:
                        delattr(node, name)
                    else:
                        setattr(node, name, previous)
            raise

    def remove_node(self, node_id: str) -> None:
        dependents = self.__dependents_in_order(node_id)
        if dependents:
            raise NodeHasDependentsError(node_id=node_id, dependents=dependents)
        self.__graph.remove_node(node=self.__graph_nodes.pop(node_id))
        node = self.nodes.pop(node_id)
        node._on_change = None  # type: ignore

    def load_workflow(self, workflow: dict[str, Any]) -> None:
        if not isinstance(workflow, dict):  # type: ignore
            reason = f"expected a dict of nodes, got {type(workflow).__name__}"
            raise InvalidWorkflowError(None, reason)
        if all(isinstance(workflow.get(key), list) for key in ("nodes", "links")):
            reason = (
                "it is in the ComfyUI UI format; "
                "export it with 'Export (API)' and load that file"
            )
            raise InvalidWorkflowError(None, reason)
        nodes = {
            node_id: _load_node(node_id, data) for node_id, data in workflow.items()
        }
        graph = IncrementalDirectedAcyclicGraph()
        graph_nodes = {
            node_id: IncrementalDirectedAcyclicGraphNode(node_id) for node_id in nodes
        }
        for graph_node in graph_nodes.values():
            graph.add_node(node=graph_node)
        for node_id, node in nodes.items():
            for name, source_id, index in _links(node):
                _check_link(nodes, node, name, source_id, index)
                try:
                    graph.add_edge(x=graph_nodes[node_id], y=graph_nodes[source_id])
                except CycleError as error:
                    cycle = " -> ".join(map(str, error.cycle))
                    reason = (
                        f"input {_comfy_name(node, name)!r} uses node {source_id!r}, "
                        f"which depends on {node_id!r}, "
                        f"so the workflow has the cycle {cycle}"
                    )
                    raise InvalidWorkflowError(node_id, reason) from error
        for node in self.nodes.values():
            node._on_change = None  # type: ignore
        self.nodes = nodes
        self.__graph = graph
        self.__graph_nodes = graph_nodes
        self.__next_index = len(nodes)
        for node_id, node in nodes.items():
            node._on_change = partial(self.__check_change, graph_nodes[node_id])  # type: ignore

    def get_comfy_ui_workflow_format(self) -> dict[str, Any]:
        workflow: dict[str, Any] = {}
        for node in self.nodes.values():
            workflow.update(node.get_comfy_ui_workflow_format())
        return workflow

    def __new_node_id(self, node_type: type[BaseNode]) -> str:
        while True:
            node_id = node_type.__name__ + f"_{self.__next_index}"
            self.__next_index += 1
            if node_id not in self.nodes:
                return node_id

    def __pipeline_outputs(self, name: str, outputs: Any) -> dict[str, tuple[str, int]]:
        if outputs is None:
            return {}
        if not isinstance(outputs, dict):
            reason = (
                "its build must return a dict of outputs or None, "
                f"not {type(outputs).__name__}"
            )
            raise PipelineError(name, reason)
        links: dict[str, tuple[str, int]] = {}
        for output, value in outputs.items():  # type: ignore
            if not isinstance(output, str):
                reason = f"its output names must be str, not {type(output).__name__}"  # type: ignore
                raise PipelineError(name, reason)
            if not _is_link(value):
                reason = (
                    f"its output {output!r} must be a link (node_id, output_index), "
                    f"not {type(value).__name__}"  # type: ignore
                )
                raise PipelineError(name, reason)
            source_id, index = value  # type: ignore
            if source_id not in self.nodes:
                reason = (
                    f"its output {output!r} uses node {source_id!r}, "
                    "which is not in the workflow"
                )
                raise PipelineError(name, reason)
            missing = _missing_output(self.nodes[source_id], index)  # type: ignore
            if missing is not None:
                raise PipelineError(name, f"its output {output!r} uses {missing}")
            links[output] = (source_id, index)
        return links

    def __check_links(self, node: BaseNode) -> None:
        for name, source_id, index in _links(node):
            if source_id not in self.nodes:
                continue
            problem = _link_problem(self.nodes[source_id], index, node, name)
            if problem is not None:
                raise InvalidLinkError(node.node_id, name, problem)

    def __graph_node(self, node_id: str) -> IncrementalDirectedAcyclicGraphNode:
        if node_id not in self.__graph_nodes:
            raise NodeNotFoundError(f"node: {node_id} not found")
        return self.__graph_nodes[node_id]

    def __dependents_in_order(self, node_id: str) -> list[str]:
        dependents = self.dependents(node_id)
        return [other_id for other_id in self.nodes if other_id in dependents]

    def __check_change(
        self, graph_node: IncrementalDirectedAcyclicGraphNode, node: BaseNode
    ) -> None:
        node_id = graph_node.data
        if node.node_id != node_id:
            dependents = self.__dependents_in_order(node_id)
            reason = (
                describe_dependents(dependents)
                if dependents
                else "a node keeps its id while it is in a workflow"
            )
            raise InvalidNodeChangeError(node_id, "node_id", reason)
        links = {name: (source_id, index) for name, source_id, index in _links(node)}
        for name, (source_id, index) in links.items():
            if source_id not in self.__graph_nodes:
                reason = f"node {source_id!r} is not in the workflow"
                raise InvalidNodeChangeError(node_id, name, reason)
            problem = _link_problem(self.nodes[source_id], index, node, name)
            if problem is not None:
                raise InvalidNodeChangeError(node_id, name, f"it would use {problem}")
        try:
            self.__link(graph_node=graph_node, node=node)
        except CycleError as error:
            source_id = error.edge[1].data
            name = next(
                name for name, (linked, _) in links.items() if linked == source_id
            )
            if source_id == node_id:
                reason = "a node cannot use its own output"
            else:
                cycle = " -> ".join(map(str, error.cycle))
                reason = (
                    f"node {source_id!r} depends on {node_id!r}, "
                    f"so the change would create the cycle {cycle}"
                )
            raise InvalidNodeChangeError(node_id, name, reason) from error

    def __link(
        self, graph_node: IncrementalDirectedAcyclicGraphNode, node: BaseNode
    ) -> None:
        dependencies = {
            self.__graph_node(value[0])
            for value in node.inputs().values()
            if _is_link(value)
        }
        for dependency in dependencies:
            self.__graph.add_edge(x=graph_node, y=dependency)
        for dependency in self.__graph.dependencies(node=graph_node) - dependencies:
            self.__graph.remove_edge(x=graph_node, y=dependency)
