from ...base import BaseNode


class CLIPTextEncodeHiDream(BaseNode):
    __slots__ = ("clip", "clip_l", "clip_g", "t5xxl", "llama")
    CATEGORY = "model/conditioning/hidream"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "clip_l": "STRING",
        "clip_g": "STRING",
        "t5xxl": "STRING",
        "llama": "STRING",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        clip_l: str | tuple[str, int],
        clip_g: str | tuple[str, int],
        t5xxl: str | tuple[str, int],
        llama: str | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.clip_l = clip_l
        self.clip_g = clip_g
        self.t5xxl = t5xxl
        self.llama = llama
