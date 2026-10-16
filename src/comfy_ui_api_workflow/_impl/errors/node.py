from ..exceptions import NodeException


def quote_names(names: list[str]) -> str:
    quoted = [repr(name) for name in names]
    if len(quoted) == 1:
        return quoted[0]
    return f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def describe_dependents(dependents: list[str]) -> str:
    if len(dependents) == 1:
        return f"node {quote_names(dependents)} depends on it"
    return f"nodes {quote_names(dependents)} depend on it"


class InvalidNodeParameterValue(NodeException):
    pass


class NodeNotHaveOutput(NodeException):
    pass


class NodeNotFoundError(NodeException):
    pass


class NodeHasDependentsError(NodeException):
    def __init__(self, node_id: str, dependents: list[str]) -> None:
        super().__init__(node_id, dependents)
        self.node_id = node_id
        self.dependents = dependents

    def __str__(self) -> str:
        return (
            f"Cannot remove node {self.node_id!r}: "
            f"{describe_dependents(self.dependents)}."
        )


class InvalidNodeChangeError(NodeException):
    def __init__(self, node_id: str, name: str, reason: str) -> None:
        super().__init__(node_id, name, reason)
        self.node_id = node_id
        self.name = name
        self.reason = reason

    def __str__(self) -> str:
        return f"Cannot change {self.name!r} of node {self.node_id!r}: {self.reason}."


class InvalidLinkError(NodeException):
    def __init__(self, node_id: str, name: str, reason: str) -> None:
        super().__init__(node_id, name, reason)
        self.node_id = node_id
        self.name = name
        self.reason = reason

    def __str__(self) -> str:
        return (
            f"Cannot add node {self.node_id!r}: input {self.name!r} uses {self.reason}."
        )
