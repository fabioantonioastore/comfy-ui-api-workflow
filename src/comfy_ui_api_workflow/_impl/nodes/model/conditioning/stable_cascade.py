from ...base import BaseNode


class StableCascade_StageB_Conditioning(BaseNode):
    __slots__ = ("conditioning", "stage_c")
    CATEGORY = "model/conditioning/stable cascade"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {"conditioning": "CONDITIONING", "stage_c": "LATENT"}

    def __init__(
        self, node_id: str, conditioning: tuple[str, int], stage_c: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.conditioning = conditioning
        self.stage_c = stage_c
