from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice, validate_range

HiDreamO1PatchSeamSmoothing_Patterns = Literal["single_shift", "symmetric"]
HiDreamO1PatchSeamSmoothing_Passes = Literal["2", "4", "ramp_2_4", "ramp_2_4_8"]
HiDreamO1PatchSeamSmoothing_Blends = Literal["average", "window", "median"]
HIDREAMO1PATCHSEAMSMOOTHING_PATTERNS = frozenset(
    get_args(HiDreamO1PatchSeamSmoothing_Patterns)
)
HIDREAMO1PATCHSEAMSMOOTHING_PASSES = frozenset(
    get_args(HiDreamO1PatchSeamSmoothing_Passes)
)
HIDREAMO1PATCHSEAMSMOOTHING_BLENDS = frozenset(
    get_args(HiDreamO1PatchSeamSmoothing_Blends)
)


class HiDreamO1PatchSeamSmoothing(BaseNode):
    __slots__ = (
        "model",
        "start_percent",
        "end_percent",
        "pattern",
        "passes",
        "blend",
        "strength",
    )
    CATEGORY = "model/patch/hidream"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
        "pattern": "COMBO",
        "passes": "COMBO",
        "blend": "COMBO",
        "strength": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        start_percent: float | tuple[str, int] = 0.8,
        end_percent: float | tuple[str, int] = 1.0,
        pattern: (
            HiDreamO1PatchSeamSmoothing_Patterns | tuple[str, int]
        ) = "single_shift",
        passes: HiDreamO1PatchSeamSmoothing_Passes | tuple[str, int] = "2",
        blend: HiDreamO1PatchSeamSmoothing_Blends | tuple[str, int] = "average",
        strength: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.pattern = pattern
        self.passes = passes
        self.blend = blend
        self.strength = strength
        self.__validate()

    def __validate(self) -> None:
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)
        validate_choice("pattern", self.pattern, HIDREAMO1PATCHSEAMSMOOTHING_PATTERNS)
        validate_choice("passes", self.passes, HIDREAMO1PATCHSEAMSMOOTHING_PASSES)
        validate_choice("blend", self.blend, HIDREAMO1PATCHSEAMSMOOTHING_BLENDS)
        validate_range("strength", self.strength, 0.0, 1.0)
