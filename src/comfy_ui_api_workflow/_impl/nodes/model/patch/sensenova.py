from ...base import BaseNode
from ...validation import validate_range


class SenseNovaSamplingOptions(BaseNode):
    __slots__ = ("model", "shift")
    CATEGORY = "model/patch/sensenova"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "shift": "FLOAT"}

    def __init__(
        self, node_id: str, model: tuple[str, int], shift: float | tuple[str, int] = 3.0
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.shift = shift
        self.__validate()

    def __validate(self) -> None:
        validate_range("shift", self.shift)
