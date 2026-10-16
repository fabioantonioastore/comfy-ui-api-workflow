from ..base import BaseNode
from ..validation import validate_range


class LTXVPreprocess(BaseNode):
    __slots__ = ("image", "img_compression")
    CATEGORY = "video/preprocessors"
    OUTPUTS = {"output_image": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"image": "IMAGE", "img_compression": "INT"}

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        img_compression: int | tuple[str, int] = 35,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.img_compression = img_compression
        self.__validate()

    def __validate(self) -> None:
        validate_range("img_compression", self.img_compression, 0, 100)
