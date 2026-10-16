from ..exceptions import PipelineException


class PipelineError(PipelineException):
    def __init__(self, pipeline: str, reason: str) -> None:
        super().__init__(pipeline, reason)
        self.pipeline = pipeline
        self.reason = reason

    def __str__(self) -> str:
        return (
            f"Cannot add pipeline {self.pipeline!r}: {self.reason}. "
            "No node of the pipeline was added."
        )
