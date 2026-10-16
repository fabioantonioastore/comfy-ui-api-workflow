from ...base import BaseNode
from ...validation import validate_range


class EmptyQwenImageLayeredLatentImage(BaseNode):
    __slots__ = ("width", "height", "layers", "batch_size")
    CATEGORY = "model/latent/qwen"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "width": "INT",
        "height": "INT",
        "layers": "INT",
        "batch_size": "INT",
    }

    def __init__(
        self,
        node_id: str,
        width: int | tuple[str, int] = 640,
        height: int | tuple[str, int] = 640,
        layers: int | tuple[str, int] = 3,
        batch_size: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.width = width
        self.height = height
        self.layers = layers
        self.batch_size = batch_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("layers", self.layers, 0, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)
