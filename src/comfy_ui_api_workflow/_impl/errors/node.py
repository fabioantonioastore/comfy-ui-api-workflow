from ..exceptions import NodeException


class InvalidNodeParameterValue(NodeException):
    pass


class NodeNotHaveOutput(NodeException):
    pass


class NodeNotFoundError(NodeException):
    pass
