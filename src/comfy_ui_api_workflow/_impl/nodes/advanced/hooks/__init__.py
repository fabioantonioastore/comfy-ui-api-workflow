from ...base import BaseNode
from ...validation import validate_range


class ConditioningTimestepsRange(BaseNode):
    __slots__ = ("start_percent", "end_percent")
    CATEGORY = "advanced/hooks"
    OUTPUTS = {"TIMESTEPS_RANGE": 0, "BEFORE_RANGE": 1, "AFTER_RANGE": 2}
    RETURN_TYPES = ("TIMESTEPS_RANGE", "TIMESTEPS_RANGE", "TIMESTEPS_RANGE")
    INPUT_SOCKET_TYPES = {"start_percent": "FLOAT", "end_percent": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        start_percent: float | tuple[str, int] = 0.0,
        end_percent: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.__validate()

    def __validate(self) -> None:
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)


from . import clip, combine, cond_pair, cond_single, create, scheduling
