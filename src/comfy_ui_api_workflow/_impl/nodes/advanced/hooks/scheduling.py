from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_bool, validate_choice, validate_range


class SetHookKeyframes(BaseNode):
    __slots__ = ("hooks", "hook_kf")
    CATEGORY = "advanced/hooks/scheduling"
    OUTPUTS = {"HOOKS": 0}
    RETURN_TYPES = ("HOOKS",)
    INPUT_SOCKET_TYPES = {"hooks": "HOOKS", "hook_kf": "HOOK_KEYFRAMES"}

    def __init__(
        self,
        node_id: str,
        hooks: tuple[str, int],
        hook_kf: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.hooks = hooks
        if hook_kf is not None:
            self.hook_kf = hook_kf


class CreateHookKeyframe(BaseNode):
    __slots__ = ("strength_mult", "start_percent", "prev_hook_kf")
    CATEGORY = "advanced/hooks/scheduling"
    OUTPUTS = {"HOOK_KF": 0}
    RETURN_TYPES = ("HOOK_KEYFRAMES",)
    INPUT_SOCKET_TYPES = {
        "strength_mult": "FLOAT",
        "start_percent": "FLOAT",
        "prev_hook_kf": "HOOK_KEYFRAMES",
    }

    def __init__(
        self,
        node_id: str,
        strength_mult: float | tuple[str, int] = 1.0,
        start_percent: float | tuple[str, int] = 0.0,
        prev_hook_kf: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.strength_mult = strength_mult
        self.start_percent = start_percent
        if prev_hook_kf is not None:
            self.prev_hook_kf = prev_hook_kf
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength_mult", self.strength_mult, -20.0, 20.0)
        validate_range("start_percent", self.start_percent, 0.0, 1.0)


CreateHookKeyframesInterpolated_Interpolations = Literal[
    "linear", "ease_in", "ease_out", "ease_in_out"
]
CREATEHOOKKEYFRAMESINTERPOLATED_INTERPOLATIONS = frozenset(
    get_args(CreateHookKeyframesInterpolated_Interpolations)
)


class CreateHookKeyframesInterpolated(BaseNode):
    __slots__ = (
        "interpolation",
        "strength_start",
        "strength_end",
        "start_percent",
        "end_percent",
        "keyframes_count",
        "print_keyframes",
        "prev_hook_kf",
    )
    CATEGORY = "advanced/hooks/scheduling"
    OUTPUTS = {"HOOK_KF": 0}
    RETURN_TYPES = ("HOOK_KEYFRAMES",)
    INPUT_SOCKET_TYPES = {
        "strength_start": "FLOAT",
        "strength_end": "FLOAT",
        "interpolation": "V1 COMBO",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
        "keyframes_count": "INT",
        "print_keyframes": "BOOLEAN",
        "prev_hook_kf": "HOOK_KEYFRAMES",
    }

    def __init__(
        self,
        node_id: str,
        interpolation: CreateHookKeyframesInterpolated_Interpolations | tuple[str, int],
        strength_start: float | tuple[str, int] = 1.0,
        strength_end: float | tuple[str, int] = 1.0,
        start_percent: float | tuple[str, int] = 0.0,
        end_percent: float | tuple[str, int] = 1.0,
        keyframes_count: int | tuple[str, int] = 5,
        print_keyframes: bool | tuple[str, int] = False,
        prev_hook_kf: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.interpolation = interpolation
        self.strength_start = strength_start
        self.strength_end = strength_end
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.keyframes_count = keyframes_count
        self.print_keyframes = print_keyframes
        if prev_hook_kf is not None:
            self.prev_hook_kf = prev_hook_kf
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "interpolation",
            self.interpolation,
            CREATEHOOKKEYFRAMESINTERPOLATED_INTERPOLATIONS,
        )
        validate_range("strength_start", self.strength_start, 0.0, 10.0)
        validate_range("strength_end", self.strength_end, 0.0, 10.0)
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)
        validate_range("keyframes_count", self.keyframes_count, 2, 100)
        validate_bool("print_keyframes", self.print_keyframes)


class CreateHookKeyframesFromFloats(BaseNode):
    __slots__ = (
        "floats_strength",
        "start_percent",
        "end_percent",
        "print_keyframes",
        "prev_hook_kf",
    )
    CATEGORY = "advanced/hooks/scheduling"
    OUTPUTS = {"HOOK_KF": 0}
    RETURN_TYPES = ("HOOK_KEYFRAMES",)
    INPUT_SOCKET_TYPES = {
        "floats_strength": "FLOATS",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
        "print_keyframes": "BOOLEAN",
        "prev_hook_kf": "HOOK_KEYFRAMES",
    }

    def __init__(
        self,
        node_id: str,
        floats_strength: tuple[str, int],
        start_percent: float | tuple[str, int] = 0.0,
        end_percent: float | tuple[str, int] = 1.0,
        print_keyframes: bool | tuple[str, int] = False,
        prev_hook_kf: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.floats_strength = floats_strength
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.print_keyframes = print_keyframes
        if prev_hook_kf is not None:
            self.prev_hook_kf = prev_hook_kf
        self.__validate()

    def __validate(self) -> None:
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)
        validate_bool("print_keyframes", self.print_keyframes)
