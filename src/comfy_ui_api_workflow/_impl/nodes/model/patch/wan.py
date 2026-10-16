from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_bool, validate_choice, validate_range

WanContextWindowsManual_Context_Schedules = Literal[
    "standard_static", "standard_uniform", "looped_uniform", "batched"
]
WanContextWindowsManual_Fuse_Methods = Literal[
    "pyramid", "relative", "flat", "overlap-linear"
]
WANCONTEXTWINDOWSMANUAL_CONTEXT_SCHEDULES = frozenset(
    get_args(WanContextWindowsManual_Context_Schedules)
)
WANCONTEXTWINDOWSMANUAL_FUSE_METHODS = frozenset(
    get_args(WanContextWindowsManual_Fuse_Methods)
)


class WanContextWindowsManual(BaseNode):
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
    CATEGORY = "model/patch/wan"
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
        context_length: int | tuple[str, int] = 81,
        context_overlap: int | tuple[str, int] = 30,
        context_schedule: (
            WanContextWindowsManual_Context_Schedules | tuple[str, int]
        ) = "standard_uniform",
        context_stride: int | tuple[str, int] = 1,
        closed_loop: bool | tuple[str, int] = False,
        fuse_method: WanContextWindowsManual_Fuse_Methods | tuple[str, int] = "pyramid",
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
            WANCONTEXTWINDOWSMANUAL_CONTEXT_SCHEDULES,
        )
        validate_range("context_stride", self.context_stride, 1)
        validate_bool("closed_loop", self.closed_loop)
        validate_choice(
            "fuse_method", self.fuse_method, WANCONTEXTWINDOWSMANUAL_FUSE_METHODS
        )
        validate_bool("freenoise", self.freenoise)
        validate_bool("retain_first_frame", self.retain_first_frame)
        validate_bool("split_conds_to_windows", self.split_conds_to_windows)


class WanUni3CControlnetApply(BaseNode):
    __slots__ = (
        "model",
        "model_patch",
        "vae",
        "render_video",
        "strength",
        "start_percent",
        "end_percent",
    )
    CATEGORY = "model/patch/wan"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "model_patch": "MODEL_PATCH",
        "vae": "VAE",
        "render_video": "IMAGE",
        "strength": "FLOAT",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        model_patch: tuple[str, int],
        vae: tuple[str, int],
        render_video: tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
        start_percent: float | tuple[str, int] = 0.0,
        end_percent: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.model_patch = model_patch
        self.vae = vae
        self.render_video = render_video
        self.strength = strength
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength", self.strength, -10.0, 10.0)
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)
