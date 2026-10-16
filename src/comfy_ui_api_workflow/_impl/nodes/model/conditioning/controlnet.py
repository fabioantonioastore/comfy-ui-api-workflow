from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice, validate_range


class ControlNetApplyAdvanced(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "control_net",
        "image",
        "strength",
        "start_percent",
        "end_percent",
        "vae",
    )
    CATEGORY = "model/conditioning/controlnet"
    OUTPUTS = {"positive": 0, "negative": 1}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "control_net": "CONTROL_NET",
        "image": "IMAGE",
        "strength": "FLOAT",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
        "vae": "VAE",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        control_net: tuple[str, int],
        image: tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
        start_percent: float | tuple[str, int] = 0.0,
        end_percent: float | tuple[str, int] = 1.0,
        vae: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.control_net = control_net
        self.image = image
        self.strength = strength
        self.start_percent = start_percent
        self.end_percent = end_percent
        if vae is not None:
            self.vae = vae
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength", self.strength, 0.0, 10.0)
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)


SetUnionControlNetType_Types = Literal[
    "auto",
    "openpose",
    "depth",
    "hed/pidi/scribble/ted",
    "canny/lineart/anime_lineart/mlsd",
    "normal",
    "segment",
    "tile",
    "repaint",
]
SETUNIONCONTROLNETTYPE_TYPES = frozenset(get_args(SetUnionControlNetType_Types))


class SetUnionControlNetType(BaseNode):
    __slots__ = ("control_net", "type")
    CATEGORY = "model/conditioning/controlnet"
    OUTPUTS = {"CONTROL_NET": 0}
    RETURN_TYPES = ("CONTROL_NET",)
    INPUT_SOCKET_TYPES = {"control_net": "CONTROL_NET", "type": "COMBO"}

    def __init__(
        self,
        node_id: str,
        control_net: tuple[str, int],
        type: SetUnionControlNetType_Types | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.control_net = control_net
        self.type = type
        self.__validate()

    def __validate(self) -> None:
        validate_choice("type", self.type, SETUNIONCONTROLNETTYPE_TYPES)


class ControlNetInpaintingAliMamaApply(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "control_net",
        "vae",
        "image",
        "mask",
        "strength",
        "start_percent",
        "end_percent",
    )
    CATEGORY = "model/conditioning/controlnet"
    OUTPUTS = {"positive": 0, "negative": 1}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "control_net": "CONTROL_NET",
        "vae": "VAE",
        "image": "IMAGE",
        "mask": "MASK",
        "strength": "FLOAT",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        control_net: tuple[str, int],
        vae: tuple[str, int],
        image: tuple[str, int],
        mask: tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
        start_percent: float | tuple[str, int] = 0.0,
        end_percent: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.control_net = control_net
        self.vae = vae
        self.image = image
        self.mask = mask
        self.strength = strength
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength", self.strength, 0.0, 10.0)
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)
