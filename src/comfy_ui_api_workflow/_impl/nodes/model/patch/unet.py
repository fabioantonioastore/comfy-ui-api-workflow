from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_bool, validate_choice, validate_range


class TomePatchModel(BaseNode):
    __slots__ = ("model", "ratio")
    CATEGORY = "model/patch/unet"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "ratio": "FLOAT"}

    def __init__(
        self, node_id: str, model: tuple[str, int], ratio: float | tuple[str, int] = 0.3
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.ratio = ratio
        self.__validate()

    def __validate(self) -> None:
        validate_range("ratio", self.ratio, 0.0, 1.0)


class FreeU(BaseNode):
    __slots__ = ("model", "b1", "b2", "s1", "s2")
    CATEGORY = "model/patch/unet"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "b1": "FLOAT",
        "b2": "FLOAT",
        "s1": "FLOAT",
        "s2": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        b1: float | tuple[str, int] = 1.1,
        b2: float | tuple[str, int] = 1.2,
        s1: float | tuple[str, int] = 0.9,
        s2: float | tuple[str, int] = 0.2,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.b1 = b1
        self.b2 = b2
        self.s1 = s1
        self.s2 = s2
        self.__validate()

    def __validate(self) -> None:
        validate_range("b1", self.b1, 0.0, 10.0)
        validate_range("b2", self.b2, 0.0, 10.0)
        validate_range("s1", self.s1, 0.0, 10.0)
        validate_range("s2", self.s2, 0.0, 10.0)


class FreeU_V2(BaseNode):
    __slots__ = ("model", "b1", "b2", "s1", "s2")
    CATEGORY = "model/patch/unet"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "b1": "FLOAT",
        "b2": "FLOAT",
        "s1": "FLOAT",
        "s2": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        b1: float | tuple[str, int] = 1.3,
        b2: float | tuple[str, int] = 1.4,
        s1: float | tuple[str, int] = 0.9,
        s2: float | tuple[str, int] = 0.2,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.b1 = b1
        self.b2 = b2
        self.s1 = s1
        self.s2 = s2
        self.__validate()

    def __validate(self) -> None:
        validate_range("b1", self.b1, 0.0, 10.0)
        validate_range("b2", self.b2, 0.0, 10.0)
        validate_range("s1", self.s1, 0.0, 10.0)
        validate_range("s2", self.s2, 0.0, 10.0)


class HyperTile(BaseNode):
    __slots__ = ("model", "tile_size", "swap_size", "max_depth", "scale_depth")
    CATEGORY = "model/patch/unet"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "tile_size": "INT",
        "swap_size": "INT",
        "max_depth": "INT",
        "scale_depth": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        tile_size: int | tuple[str, int] = 256,
        swap_size: int | tuple[str, int] = 2,
        max_depth: int | tuple[str, int] = 0,
        scale_depth: bool | tuple[str, int] = False,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.tile_size = tile_size
        self.swap_size = swap_size
        self.max_depth = max_depth
        self.scale_depth = scale_depth
        self.__validate()

    def __validate(self) -> None:
        validate_range("tile_size", self.tile_size, 1, 2048)
        validate_range("swap_size", self.swap_size, 1, 128)
        validate_range("max_depth", self.max_depth, 0, 10)
        validate_bool("scale_depth", self.scale_depth)


PatchModelAddDownscale_Downscale_Methods = Literal[
    "bicubic", "nearest-exact", "bilinear", "area", "bislerp"
]
PatchModelAddDownscale_Upscale_Methods = Literal[
    "bicubic", "nearest-exact", "bilinear", "area", "bislerp"
]
PATCHMODELADDDOWNSCALE_DOWNSCALE_METHODS = frozenset(
    get_args(PatchModelAddDownscale_Downscale_Methods)
)
PATCHMODELADDDOWNSCALE_UPSCALE_METHODS = frozenset(
    get_args(PatchModelAddDownscale_Upscale_Methods)
)


class PatchModelAddDownscale(BaseNode):
    __slots__ = (
        "model",
        "downscale_method",
        "upscale_method",
        "block_number",
        "downscale_factor",
        "start_percent",
        "end_percent",
        "downscale_after_skip",
    )
    CATEGORY = "model/patch/unet"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "block_number": "INT",
        "downscale_factor": "FLOAT",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
        "downscale_after_skip": "BOOLEAN",
        "downscale_method": "COMBO",
        "upscale_method": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        downscale_method: PatchModelAddDownscale_Downscale_Methods | tuple[str, int],
        upscale_method: PatchModelAddDownscale_Upscale_Methods | tuple[str, int],
        block_number: int | tuple[str, int] = 3,
        downscale_factor: float | tuple[str, int] = 2.0,
        start_percent: float | tuple[str, int] = 0.0,
        end_percent: float | tuple[str, int] = 0.35,
        downscale_after_skip: bool | tuple[str, int] = True,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.downscale_method = downscale_method
        self.upscale_method = upscale_method
        self.block_number = block_number
        self.downscale_factor = downscale_factor
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.downscale_after_skip = downscale_after_skip
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "downscale_method",
            self.downscale_method,
            PATCHMODELADDDOWNSCALE_DOWNSCALE_METHODS,
        )
        validate_choice(
            "upscale_method",
            self.upscale_method,
            PATCHMODELADDDOWNSCALE_UPSCALE_METHODS,
        )
        validate_range("block_number", self.block_number, 1, 32)
        validate_range("downscale_factor", self.downscale_factor, 0.1, 9.0)
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)
        validate_bool("downscale_after_skip", self.downscale_after_skip)


class PerturbedAttentionGuidance(BaseNode):
    __slots__ = ("model", "scale")
    CATEGORY = "model/patch/unet"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "scale": "FLOAT"}

    def __init__(
        self, node_id: str, model: tuple[str, int], scale: float | tuple[str, int] = 3.0
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.scale = scale
        self.__validate()

    def __validate(self) -> None:
        validate_range("scale", self.scale, 0.0, 100.0)


class EpsilonScaling(BaseNode):
    __slots__ = ("model", "scaling_factor")
    CATEGORY = "model/patch/unet"
    CLASS_TYPE = "Epsilon Scaling"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "scaling_factor": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        scaling_factor: float | tuple[str, int] = 1.005,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.scaling_factor = scaling_factor
        self.__validate()

    def __validate(self) -> None:
        validate_range("scaling_factor", self.scaling_factor, 0.5, 1.5)


class TemporalScoreRescaling(BaseNode):
    __slots__ = ("model", "tsr_k", "tsr_sigma")
    CATEGORY = "model/patch/unet"
    OUTPUTS = {"patched_model": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "tsr_k": "FLOAT", "tsr_sigma": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        tsr_k: float | tuple[str, int] = 0.95,
        tsr_sigma: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.tsr_k = tsr_k
        self.tsr_sigma = tsr_sigma
        self.__validate()

    def __validate(self) -> None:
        validate_range("tsr_k", self.tsr_k, 0.01, 100.0)
        validate_range("tsr_sigma", self.tsr_sigma, 0.01, 100.0)
