from ...base import BaseNode
from ...validation import validate_range


class ConditioningStableAudio(BaseNode):
    __slots__ = ("positive", "negative", "seconds_start", "seconds_total")
    CATEGORY = "model/conditioning/stable audio"
    OUTPUTS = {"positive": 0, "negative": 1}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "seconds_start": "FLOAT",
        "seconds_total": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        seconds_start: float | tuple[str, int] = 0.0,
        seconds_total: float | tuple[str, int] = 47.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.seconds_start = seconds_start
        self.seconds_total = seconds_total
        self.__validate()

    def __validate(self) -> None:
        validate_range("seconds_start", self.seconds_start, 0.0, 1000.0)
        validate_range("seconds_total", self.seconds_total, 0.0, 1000.0)
