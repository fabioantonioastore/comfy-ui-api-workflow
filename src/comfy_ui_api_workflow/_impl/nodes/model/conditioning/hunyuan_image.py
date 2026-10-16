from ...base import BaseNode


class CLIPTextEncodeHunyuanDiT(BaseNode):
    __slots__ = ("clip", "bert", "mt5xl")
    CATEGORY = "model/conditioning/hunyuan image"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {"clip": "CLIP", "bert": "STRING", "mt5xl": "STRING"}

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        bert: str | tuple[str, int],
        mt5xl: str | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.bert = bert
        self.mt5xl = mt5xl
