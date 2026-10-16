from ...base import BaseNode
from ...validation import validate_range


class LumaReferenceNode(BaseNode):
    __slots__ = ("image", "weight", "luma_ref")
    CATEGORY = "partner/image/Luma"
    OUTPUTS = {"luma_ref": 0}
    RETURN_TYPES = ("LUMA_REF",)
    INPUT_SOCKET_TYPES = {"image": "IMAGE", "weight": "FLOAT", "luma_ref": "LUMA_REF"}

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        weight: float | tuple[str, int] = 1.0,
        luma_ref: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.weight = weight
        if luma_ref is not None:
            self.luma_ref = luma_ref
        self.__validate()

    def __validate(self) -> None:
        validate_range("weight", self.weight, 0.0, 1.0)
