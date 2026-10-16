from ...base import BaseNode
from ...validation import validate_range


class GLIGENTextBoxApply(BaseNode):
    __slots__ = (
        "conditioning_to",
        "clip",
        "gligen_textbox_model",
        "text",
        "width",
        "height",
        "x",
        "y",
    )
    CATEGORY = "model/conditioning/gligen"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "conditioning_to": "CONDITIONING",
        "clip": "CLIP",
        "gligen_textbox_model": "GLIGEN",
        "text": "STRING",
        "width": "INT",
        "height": "INT",
        "x": "INT",
        "y": "INT",
    }

    def __init__(
        self,
        node_id: str,
        conditioning_to: tuple[str, int],
        clip: tuple[str, int],
        gligen_textbox_model: tuple[str, int],
        text: str | tuple[str, int],
        width: int | tuple[str, int] = 64,
        height: int | tuple[str, int] = 64,
        x: int | tuple[str, int] = 0,
        y: int | tuple[str, int] = 0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.conditioning_to = conditioning_to
        self.clip = clip
        self.gligen_textbox_model = gligen_textbox_model
        self.text = text
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 8, 16_384)
        validate_range("height", self.height, 8, 16_384)
        validate_range("x", self.x, 0, 16_384)
        validate_range("y", self.y, 0, 16_384)
