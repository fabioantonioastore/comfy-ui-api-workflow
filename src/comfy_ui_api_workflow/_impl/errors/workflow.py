from ..exceptions import WorkflowException


class InvalidWorkflowError(WorkflowException):
    def __init__(self, node_id: str | None, reason: str) -> None:
        super().__init__(node_id, reason)
        self.node_id = node_id
        self.reason = reason

    def __str__(self) -> str:
        if self.node_id is None:
            return f"Cannot load the workflow: {self.reason}."
        return f"Cannot load node {self.node_id!r}: {self.reason}."
