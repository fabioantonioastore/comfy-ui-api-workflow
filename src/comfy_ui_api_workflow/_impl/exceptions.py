class ComfyUIAPIWorkflowException(Exception):
    pass


class DataStructureException(ComfyUIAPIWorkflowException):
    pass


class WorkflowException(ComfyUIAPIWorkflowException):
    pass


class NodeException(ComfyUIAPIWorkflowException):
    pass


class PipelineException(ComfyUIAPIWorkflowException):
    pass


class OutputException(ComfyUIAPIWorkflowException):
    pass
