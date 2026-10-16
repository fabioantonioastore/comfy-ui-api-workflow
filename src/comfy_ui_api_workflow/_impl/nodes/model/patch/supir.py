from ...base import BaseNode
from ...validation import validate_range


class SUPIRApply(BaseNode):
    __slots__ = (
        "model",
        "model_patch",
        "vae",
        "image",
        "strength_start",
        "strength_end",
        "restore_cfg",
        "restore_cfg_s_tmin",
    )
    CATEGORY = "model/patch/supir"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "model_patch": "MODEL_PATCH",
        "vae": "VAE",
        "image": "IMAGE",
        "strength_start": "FLOAT",
        "strength_end": "FLOAT",
        "restore_cfg": "FLOAT",
        "restore_cfg_s_tmin": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        model_patch: tuple[str, int],
        vae: tuple[str, int],
        image: tuple[str, int],
        strength_start: float | tuple[str, int] = 1.0,
        strength_end: float | tuple[str, int] = 1.0,
        restore_cfg: float | tuple[str, int] = 4.0,
        restore_cfg_s_tmin: float | tuple[str, int] = 0.05,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.model_patch = model_patch
        self.vae = vae
        self.image = image
        self.strength_start = strength_start
        self.strength_end = strength_end
        self.restore_cfg = restore_cfg
        self.restore_cfg_s_tmin = restore_cfg_s_tmin
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength_start", self.strength_start, 0.0, 10.0)
        validate_range("strength_end", self.strength_end, 0.0, 10.0)
        validate_range("restore_cfg", self.restore_cfg, 0.0, 20.0)
        validate_range("restore_cfg_s_tmin", self.restore_cfg_s_tmin, 0.0, 1.0)
