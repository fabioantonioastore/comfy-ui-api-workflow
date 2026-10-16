from ...base import BaseNode
from ...validation import validate_range


class AnimaLLLiteApply(BaseNode):
    __slots__ = (
        "model",
        "model_patch",
        "image",
        "strength",
        "start_percent",
        "end_percent",
        "mask",
    )
    CATEGORY = "model/patch/anima"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "model_patch": "MODEL_PATCH",
        "image": "IMAGE",
        "strength": "FLOAT",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
        "mask": "MASK",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        model_patch: tuple[str, int],
        image: tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
        start_percent: float | tuple[str, int] = 0.0,
        end_percent: float | tuple[str, int] = 1.0,
        mask: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.model_patch = model_patch
        self.image = image
        self.strength = strength
        self.start_percent = start_percent
        self.end_percent = end_percent
        if mask is not None:
            self.mask = mask
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength", self.strength, -10.0, 10.0)
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)
