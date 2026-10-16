from ..base import BaseNode
from ..validation import validate_range


class AdjustBrightness(BaseNode):
    __slots__ = ("images", "factor")
    CATEGORY = "image/adjustments"
    OUTPUTS = {"images": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"images": "IMAGE", "factor": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        factor: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.factor = factor
        self.__validate()

    def __validate(self) -> None:
        validate_range("factor", self.factor, 0.0, 2.0)


class AdjustContrast(BaseNode):
    __slots__ = ("images", "factor")
    CATEGORY = "image/adjustments"
    OUTPUTS = {"images": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"images": "IMAGE", "factor": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        factor: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.factor = factor
        self.__validate()

    def __validate(self) -> None:
        validate_range("factor", self.factor, 0.0, 2.0)
