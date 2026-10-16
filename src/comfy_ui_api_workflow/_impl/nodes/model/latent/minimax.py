from ...base import BaseNode
from ...validation import validate_range


class EmptyMiniMaxH3LatentAV(BaseNode):
    __slots__ = ("width", "height", "length")
    CATEGORY = "model/latent/minimax"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"width": "INT", "height": "INT", "length": "INT"}

    def __init__(
        self,
        node_id: str,
        width: int | tuple[str, int] = 1344,
        height: int | tuple[str, int] = 768,
        length: int | tuple[str, int] = 124,
    ) -> None:
        super().__init__(node_id=node_id)
        self.width = width
        self.height = height
        self.length = length
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 32, 16_384)
        validate_range("height", self.height, 32, 16_384)
        validate_range("length", self.length, 5, 3600)
