from ...base import BaseNode
from ...validation import validate_range


class Krea2StyleReferenceNode(BaseNode):
    __slots__ = ("image", "strength", "style_reference")
    CATEGORY = "partner/image/Krea"
    OUTPUTS = {"style_reference": 0}
    RETURN_TYPES = ("KREA_STYLE_REF",)
    INPUT_SOCKET_TYPES = {
        "image": "IMAGE",
        "strength": "FLOAT",
        "style_reference": "KREA_STYLE_REF",
    }

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
        style_reference: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.strength = strength
        if style_reference is not None:
            self.style_reference = style_reference
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength", self.strength, -2.0, 2.0)
