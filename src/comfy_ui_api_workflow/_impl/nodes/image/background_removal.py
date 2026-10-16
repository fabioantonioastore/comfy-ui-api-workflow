from ..base import BaseNode


class RemoveBackground(BaseNode):
    __slots__ = ("bg_removal_model", "image")
    CATEGORY = "image/background removal"
    OUTPUTS = {"mask": 0}
    RETURN_TYPES = ("MASK",)
    INPUT_SOCKET_TYPES = {"bg_removal_model": "BACKGROUND_REMOVAL", "image": "IMAGE"}

    def __init__(
        self, node_id: str, bg_removal_model: tuple[str, int], image: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.bg_removal_model = bg_removal_model
        self.image = image
