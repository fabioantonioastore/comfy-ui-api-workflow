from ...base import BaseNode
from ...validation import validate_range


class MiniMaxH3SigmaShift(BaseNode):
    __slots__ = ("model", "shift_video", "shift_audio")
    CATEGORY = "model/patch/minimax"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "shift_video": "FLOAT",
        "shift_audio": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        shift_video: float | tuple[str, int] = 12.0,
        shift_audio: float | tuple[str, int] = 3.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.shift_video = shift_video
        self.shift_audio = shift_audio
        self.__validate()

    def __validate(self) -> None:
        validate_range("shift_video", self.shift_video, 0.01, 100.0)
        validate_range("shift_audio", self.shift_audio, 0.01, 100.0)


class MiniMaxH3FunControlNetApply(BaseNode):
    __slots__ = (
        "model",
        "model_patch",
        "vae",
        "strength",
        "start_percent",
        "end_percent",
        "control_video",
        "mask",
        "source_video",
    )
    CATEGORY = "model/patch/minimax"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "model_patch": "MODEL_PATCH",
        "vae": "VAE",
        "strength": "FLOAT",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
        "control_video": "IMAGE",
        "mask": "MASK",
        "source_video": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        model_patch: tuple[str, int],
        vae: tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
        start_percent: float | tuple[str, int] = 0.0,
        end_percent: float | tuple[str, int] = 1.0,
        control_video: tuple[str, int] | None = None,
        mask: tuple[str, int] | None = None,
        source_video: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.model_patch = model_patch
        self.vae = vae
        self.strength = strength
        self.start_percent = start_percent
        self.end_percent = end_percent
        if control_video is not None:
            self.control_video = control_video
        if mask is not None:
            self.mask = mask
        if source_video is not None:
            self.source_video = source_video
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength", self.strength, 0.0, 10.0)
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)
