from ...base import BaseNode
from ...validation import validate_range


class ModelSamplingFlux(BaseNode):
    __slots__ = ("model", "max_shift", "base_shift", "width", "height")
    CATEGORY = "model/patch/flux"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "max_shift": "FLOAT",
        "base_shift": "FLOAT",
        "width": "INT",
        "height": "INT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        max_shift: float | tuple[str, int] = 1.15,
        base_shift: float | tuple[str, int] = 0.5,
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.max_shift = max_shift
        self.base_shift = base_shift
        self.width = width
        self.height = height
        self.__validate()

    def __validate(self) -> None:
        validate_range("max_shift", self.max_shift, 0.0, 100.0)
        validate_range("base_shift", self.base_shift, 0.0, 100.0)
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)


class USOStyleReference(BaseNode):
    __slots__ = ("model", "model_patch", "clip_vision_output")
    CATEGORY = "model/patch/flux"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "model_patch": "MODEL_PATCH",
        "clip_vision_output": "CLIP_VISION_OUTPUT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        model_patch: tuple[str, int],
        clip_vision_output: tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.model_patch = model_patch
        self.clip_vision_output = clip_vision_output
