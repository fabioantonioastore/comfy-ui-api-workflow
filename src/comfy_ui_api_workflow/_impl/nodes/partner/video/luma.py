from typing import Literal

from ...base import BaseNode
from ...dynamic import Combo, DynamicInputs, Number, validate_dynamic

LumaRay32KeyframeNode_Positions = Literal[
    "Fraction of duration (0.0-1.0)", "Absolute time (seconds)"
]


class LumaRay32KeyframeNode(DynamicInputs, BaseNode):
    __slots__ = (
        "image",
        "position",
        "position_fraction",
        "position_seconds",
        "keyframes",
    )
    CATEGORY = "partner/video/Luma"
    OUTPUTS = {"keyframes": 0}
    RETURN_TYPES = ("LUMA_RAY32_KEYFRAME",)
    INPUT_NAMES = {
        "position_fraction": "position.fraction",
        "position_seconds": "position.seconds",
    }
    INPUT_SOCKET_TYPES = {
        "image": "IMAGE",
        "position": "COMFY_DYNAMICCOMBO_V3",
        "position_fraction": "FLOAT",
        "position_seconds": "FLOAT",
        "keyframes": "LUMA_RAY32_KEYFRAME",
    }
    DYNAMIC_INPUTS = {
        "position": Combo(
            {
                "Fraction of duration (0.0-1.0)": {
                    "position_fraction": Number(default=0.0, low=0.0, high=1.0)
                },
                "Absolute time (seconds)": {
                    "position_seconds": Number(default=0.0, low=0.0, high=10.0)
                },
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        position: LumaRay32KeyframeNode_Positions = "Fraction of duration (0.0-1.0)",
        position_fraction: float | tuple[str, int] | None = None,
        position_seconds: float | tuple[str, int] | None = None,
        keyframes: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.position = position
        if position_fraction is not None:
            self.position_fraction = position_fraction
        if position_seconds is not None:
            self.position_seconds = position_seconds
        if keyframes is not None:
            self.keyframes = keyframes
        self.__validate()

    def __validate(self) -> None:
        validate_dynamic(self)
