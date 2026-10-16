from ...base import BaseNode


class PhotoMakerEncode(BaseNode):
    __slots__ = ("photomaker", "image", "clip", "text")
    CATEGORY = "model/conditioning/photomaker"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "photomaker": "PHOTOMAKER",
        "image": "IMAGE",
        "clip": "CLIP",
        "text": "STRING",
    }

    def __init__(
        self,
        node_id: str,
        photomaker: tuple[str, int],
        image: tuple[str, int],
        clip: tuple[str, int],
        text: str | tuple[str, int] = "photograph of photomaker",
    ) -> None:
        super().__init__(node_id=node_id)
        self.photomaker = photomaker
        self.image = image
        self.clip = clip
        self.text = text
