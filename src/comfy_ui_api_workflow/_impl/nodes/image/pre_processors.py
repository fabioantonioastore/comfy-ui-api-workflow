from ..base import BaseNode


class SeedVR2Preprocess(BaseNode):
    __slots__ = ("resized_images",)
    CATEGORY = "image/pre-processors"
    OUTPUTS = {"images": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"resized_images": "IMAGE"}

    def __init__(self, node_id: str, resized_images: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.resized_images = resized_images
