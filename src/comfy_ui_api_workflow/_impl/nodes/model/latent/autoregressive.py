from ...base import BaseNode
from ...validation import validate_range


class EmptyARVideoLatent(BaseNode):
    __slots__ = ("width", "height", "length", "batch_size")
    CATEGORY = "model/latent/autoregressive"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
    }

    def __init__(
        self,
        node_id: str,
        width: int | tuple[str, int] = 832,
        height: int | tuple[str, int] = 480,
        length: int | tuple[str, int] = 81,
        batch_size: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 8192)
        validate_range("height", self.height, 16, 8192)
        validate_range("length", self.length, 1, 1024)
        validate_range("batch_size", self.batch_size, 1, 64)
