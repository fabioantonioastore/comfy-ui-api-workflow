from typing import Literal

from ...base import BaseNode
from ...dynamic import Combo, DynamicInputs, Number, validate_dynamic

RunwayAleph2KeyframeNode_Timings = Literal[
    "Absolute time (seconds)", "Fraction of duration (0.0-1.0)"
]


class RunwayAleph2KeyframeNode(DynamicInputs, BaseNode):
    __slots__ = (
        "image",
        "timing",
        "timing_seconds",
        "timing_fraction",
        "keyframes",
    )
    CATEGORY = "partner/video/Runway"
    OUTPUTS = {"keyframes": 0}
    RETURN_TYPES = ("RUNWAY_ALEPH2_KEYFRAME",)
    INPUT_NAMES = {
        "timing_seconds": "timing.seconds",
        "timing_fraction": "timing.fraction",
    }
    INPUT_SOCKET_TYPES = {
        "image": "IMAGE",
        "timing": "COMFY_DYNAMICCOMBO_V3",
        "timing_seconds": "FLOAT",
        "timing_fraction": "FLOAT",
        "keyframes": "RUNWAY_ALEPH2_KEYFRAME",
    }
    DYNAMIC_INPUTS = {
        "timing": Combo(
            {
                "Absolute time (seconds)": {
                    "timing_seconds": Number(default=0.0, low=0.0, high=30.0)
                },
                "Fraction of duration (0.0-1.0)": {
                    "timing_fraction": Number(default=0.0, low=0.0, high=1.0)
                },
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        timing: RunwayAleph2KeyframeNode_Timings = "Absolute time (seconds)",
        timing_seconds: float | tuple[str, int] | None = None,
        timing_fraction: float | tuple[str, int] | None = None,
        keyframes: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.timing = timing
        if timing_seconds is not None:
            self.timing_seconds = timing_seconds
        if timing_fraction is not None:
            self.timing_fraction = timing_fraction
        if keyframes is not None:
            self.keyframes = keyframes
        self.__validate()

    def __validate(self) -> None:
        validate_dynamic(self)


RunwayAleph2PromptImageNode_Positions = Literal[
    "Absolute time (seconds)", "Fraction of duration (0.0-1.0)"
]


class RunwayAleph2PromptImageNode(DynamicInputs, BaseNode):
    __slots__ = (
        "image",
        "position",
        "position_seconds",
        "position_fraction",
        "prompt_images",
    )
    CATEGORY = "partner/video/Runway"
    OUTPUTS = {"prompt_images": 0}
    RETURN_TYPES = ("RUNWAY_ALEPH2_PROMPT_IMAGE",)
    INPUT_NAMES = {
        "position_seconds": "position.seconds",
        "position_fraction": "position.fraction",
    }
    INPUT_SOCKET_TYPES = {
        "image": "IMAGE",
        "position": "COMFY_DYNAMICCOMBO_V3",
        "position_seconds": "FLOAT",
        "position_fraction": "FLOAT",
        "prompt_images": "RUNWAY_ALEPH2_PROMPT_IMAGE",
    }
    DYNAMIC_INPUTS = {
        "position": Combo(
            {
                "Absolute time (seconds)": {
                    "position_seconds": Number(default=0.0, low=0.0, high=30.0)
                },
                "Fraction of duration (0.0-1.0)": {
                    "position_fraction": Number(default=0.0, low=0.0, high=1.0)
                },
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        position: RunwayAleph2PromptImageNode_Positions = "Absolute time (seconds)",
        position_seconds: float | tuple[str, int] | None = None,
        position_fraction: float | tuple[str, int] | None = None,
        prompt_images: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.position = position
        if position_seconds is not None:
            self.position_seconds = position_seconds
        if position_fraction is not None:
            self.position_fraction = position_fraction
        if prompt_images is not None:
            self.prompt_images = prompt_images
        self.__validate()

    def __validate(self) -> None:
        validate_dynamic(self)
