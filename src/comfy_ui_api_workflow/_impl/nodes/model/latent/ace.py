from ...base import BaseNode
from ...validation import validate_range


class EmptyAceStepLatentAudio(BaseNode):
    __slots__ = ("seconds", "batch_size")
    CATEGORY = "model/latent/ace"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"seconds": "FLOAT", "batch_size": "INT"}

    def __init__(
        self,
        node_id: str,
        seconds: float | tuple[str, int] = 120.0,
        batch_size: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.seconds = seconds
        self.batch_size = batch_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("seconds", self.seconds, 1.0, 1000.0)
        validate_range("batch_size", self.batch_size, 1, 4096)


class EmptyAceStep1_5LatentAudio(BaseNode):
    __slots__ = ("seconds", "batch_size")
    CATEGORY = "model/latent/ace"
    CLASS_TYPE = "EmptyAceStep1.5LatentAudio"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"seconds": "FLOAT", "batch_size": "INT"}

    def __init__(
        self,
        node_id: str,
        seconds: float | tuple[str, int] = 120.0,
        batch_size: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.seconds = seconds
        self.batch_size = batch_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("seconds", self.seconds, 1.0, 1000.0)
        validate_range("batch_size", self.batch_size, 1, 4096)
