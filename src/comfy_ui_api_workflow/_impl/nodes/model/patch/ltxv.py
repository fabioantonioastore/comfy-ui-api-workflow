from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_bool, validate_choice, validate_range


class ModelSamplingLTXV(BaseNode):
    __slots__ = ("model", "max_shift", "base_shift", "latent")
    CATEGORY = "model/patch/ltxv"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "max_shift": "FLOAT",
        "base_shift": "FLOAT",
        "latent": "LATENT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        max_shift: float | tuple[str, int] = 2.05,
        base_shift: float | tuple[str, int] = 0.95,
        latent: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.max_shift = max_shift
        self.base_shift = base_shift
        if latent is not None:
            self.latent = latent
        self.__validate()

    def __validate(self) -> None:
        validate_range("max_shift", self.max_shift, 0.0, 100.0)
        validate_range("base_shift", self.base_shift, 0.0, 100.0)


LTXVContextWindows_Context_Schedules = Literal[
    "standard_static", "standard_uniform", "looped_uniform", "batched"
]
LTXVContextWindows_Fuse_Methods = Literal[
    "pyramid", "relative", "flat", "overlap-linear"
]
LTXVCONTEXTWINDOWS_CONTEXT_SCHEDULES = frozenset(
    get_args(LTXVContextWindows_Context_Schedules)
)
LTXVCONTEXTWINDOWS_FUSE_METHODS = frozenset(get_args(LTXVContextWindows_Fuse_Methods))


class LTXVContextWindows(BaseNode):
    __slots__ = (
        "model",
        "context_length",
        "context_overlap",
        "context_schedule",
        "context_stride",
        "closed_loop",
        "fuse_method",
        "freenoise",
        "retain_first_frame",
        "split_conds_to_windows",
    )
    CATEGORY = "model/patch/ltxv"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "context_length": "INT",
        "context_overlap": "INT",
        "context_schedule": "COMBO",
        "context_stride": "INT",
        "closed_loop": "BOOLEAN",
        "fuse_method": "COMBO",
        "freenoise": "BOOLEAN",
        "retain_first_frame": "BOOLEAN",
        "split_conds_to_windows": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        context_length: int | tuple[str, int] = 145,
        context_overlap: int | tuple[str, int] = 40,
        context_schedule: (
            LTXVContextWindows_Context_Schedules | tuple[str, int]
        ) = "standard_uniform",
        context_stride: int | tuple[str, int] = 1,
        closed_loop: bool | tuple[str, int] = False,
        fuse_method: LTXVContextWindows_Fuse_Methods | tuple[str, int] = "pyramid",
        freenoise: bool | tuple[str, int] = True,
        retain_first_frame: bool | tuple[str, int] = False,
        split_conds_to_windows: bool | tuple[str, int] = False,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.context_length = context_length
        self.context_overlap = context_overlap
        self.context_schedule = context_schedule
        self.context_stride = context_stride
        self.closed_loop = closed_loop
        self.fuse_method = fuse_method
        self.freenoise = freenoise
        self.retain_first_frame = retain_first_frame
        self.split_conds_to_windows = split_conds_to_windows
        self.__validate()

    def __validate(self) -> None:
        validate_range("context_length", self.context_length, 1, 16_384)
        validate_range("context_overlap", self.context_overlap, 0)
        validate_choice(
            "context_schedule",
            self.context_schedule,
            LTXVCONTEXTWINDOWS_CONTEXT_SCHEDULES,
        )
        validate_range("context_stride", self.context_stride, 1)
        validate_bool("closed_loop", self.closed_loop)
        validate_choice(
            "fuse_method", self.fuse_method, LTXVCONTEXTWINDOWS_FUSE_METHODS
        )
        validate_bool("freenoise", self.freenoise)
        validate_bool("retain_first_frame", self.retain_first_frame)
        validate_bool("split_conds_to_windows", self.split_conds_to_windows)
