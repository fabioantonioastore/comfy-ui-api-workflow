from ...base import BaseNode
from ...validation import validate_range


class CLIPTextEncodePixArtAlpha(BaseNode):
    __slots__ = ("text", "clip", "width", "height")
    CATEGORY = "model/conditioning/pixart"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "width": "INT",
        "height": "INT",
        "text": "STRING",
        "clip": "CLIP",
    }

    def __init__(
        self,
        node_id: str,
        text: str | tuple[str, int],
        clip: tuple[str, int],
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
    ) -> None:
        super().__init__(node_id=node_id)
        self.text = text
        self.clip = clip
        self.width = width
        self.height = height
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 0, 16_384)
        validate_range("height", self.height, 0, 16_384)
