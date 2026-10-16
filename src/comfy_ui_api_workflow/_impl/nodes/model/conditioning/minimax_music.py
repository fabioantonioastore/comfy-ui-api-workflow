from ...base import BaseNode
from ...validation import validate_range


class MiniMaxMusic3TextEncode(BaseNode):
    __slots__ = (
        "clip",
        "caption",
        "lyrics",
        "seed",
        "max_duration",
        "cfg_scale",
        "top_k",
    )
    CATEGORY = "model/conditioning/minimax music"
    OUTPUTS = {"CONDITIONING": 0, "seconds": 1}
    RETURN_TYPES = ("CONDITIONING", "FLOAT")
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "caption": "STRING",
        "lyrics": "STRING",
        "seed": "INT",
        "max_duration": "FLOAT",
        "cfg_scale": "FLOAT",
        "top_k": "INT",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        caption: str | tuple[str, int],
        lyrics: str | tuple[str, int],
        seed: int | tuple[str, int] = 0,
        max_duration: float | tuple[str, int] = 120.0,
        cfg_scale: float | tuple[str, int] = 1.5,
        top_k: int | tuple[str, int] = 50,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.caption = caption
        self.lyrics = lyrics
        self.seed = seed
        self.max_duration = max_duration
        self.cfg_scale = cfg_scale
        self.top_k = top_k
        self.__validate()

    def __validate(self) -> None:
        validate_range("seed", self.seed, 0, 18_446_744_073_709_551_615)
        validate_range("max_duration", self.max_duration, 0.04, 360.0)
        validate_range("cfg_scale", self.cfg_scale, 0.0, 100.0)
        validate_range("top_k", self.top_k, 1, 16_384)
