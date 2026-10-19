from typing import Literal, get_args

from ..base import BaseNode
from ..validation import validate_choice, validate_range
from ..dynamic import Combo, DynamicInputs, Number, validate_dynamic

ImageBlend_Blend_Modes = Literal[
    "normal", "multiply", "screen", "overlay", "soft_light", "difference"
]
IMAGEBLEND_BLEND_MODES = frozenset(get_args(ImageBlend_Blend_Modes))


class ImageBlend(BaseNode):
    __slots__ = ("image1", "image2", "blend_mode", "blend_factor")
    CATEGORY = "image/filters"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "image1": "IMAGE",
        "image2": "IMAGE",
        "blend_factor": "FLOAT",
        "blend_mode": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        image1: tuple[str, int],
        image2: tuple[str, int],
        blend_mode: ImageBlend_Blend_Modes | tuple[str, int],
        blend_factor: float | tuple[str, int] = 0.5,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image1 = image1
        self.image2 = image2
        self.blend_mode = blend_mode
        self.blend_factor = blend_factor
        self.__validate()

    def __validate(self) -> None:
        validate_choice("blend_mode", self.blend_mode, IMAGEBLEND_BLEND_MODES)
        validate_range("blend_factor", self.blend_factor, 0.0, 1.0)


class ImageBlur(BaseNode):
    __slots__ = ("image", "blur_radius", "sigma")
    CATEGORY = "image/filters"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"image": "IMAGE", "blur_radius": "INT", "sigma": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        blur_radius: int | tuple[str, int] = 1,
        sigma: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.blur_radius = blur_radius
        self.sigma = sigma
        self.__validate()

    def __validate(self) -> None:
        validate_range("blur_radius", self.blur_radius, 1, 31)
        validate_range("sigma", self.sigma, 0.1, 10.0)


ImageQuantize_Dithers = Literal[
    "none", "floyd-steinberg", "bayer-2", "bayer-4", "bayer-8", "bayer-16"
]
IMAGEQUANTIZE_DITHERS = frozenset(get_args(ImageQuantize_Dithers))


class ImageQuantize(BaseNode):
    __slots__ = ("image", "dither", "colors")
    CATEGORY = "image/filters"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"image": "IMAGE", "colors": "INT", "dither": "COMBO"}

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        dither: ImageQuantize_Dithers | tuple[str, int],
        colors: int | tuple[str, int] = 256,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.dither = dither
        self.colors = colors
        self.__validate()

    def __validate(self) -> None:
        validate_choice("dither", self.dither, IMAGEQUANTIZE_DITHERS)
        validate_range("colors", self.colors, 1, 256)


class ImageSharpen(BaseNode):
    __slots__ = ("image", "sharpen_radius", "sigma", "alpha")
    CATEGORY = "image/filters"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "image": "IMAGE",
        "sharpen_radius": "INT",
        "sigma": "FLOAT",
        "alpha": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        sharpen_radius: int | tuple[str, int] = 1,
        sigma: float | tuple[str, int] = 1.0,
        alpha: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.sharpen_radius = sharpen_radius
        self.sigma = sigma
        self.alpha = alpha
        self.__validate()

    def __validate(self) -> None:
        validate_range("sharpen_radius", self.sharpen_radius, 1, 31)
        validate_range("sigma", self.sigma, 0.1, 10.0)
        validate_range("alpha", self.alpha, 0.0, 5.0)


class Canny(BaseNode):
    __slots__ = ("image", "low_threshold", "high_threshold")
    CATEGORY = "image/filters"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "image": "IMAGE",
        "low_threshold": "FLOAT",
        "high_threshold": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        low_threshold: float | tuple[str, int] = 0.4,
        high_threshold: float | tuple[str, int] = 0.8,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.__validate()

    def __validate(self) -> None:
        validate_range("low_threshold", self.low_threshold, 0.01, 0.99)
        validate_range("high_threshold", self.high_threshold, 0.01, 0.99)


class ImageAddNoise(BaseNode):
    __slots__ = ("image", "seed", "strength")
    CATEGORY = "image/filters"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"image": "IMAGE", "seed": "INT", "strength": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        seed: int | tuple[str, int] = 0,
        strength: float | tuple[str, int] = 0.5,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.seed = seed
        self.strength = strength
        self.__validate()

    def __validate(self) -> None:
        validate_range("seed", self.seed, 0, 18_446_744_073_709_551_615)
        validate_range("strength", self.strength, 0.0, 1.0)


Morphology_Operations = Literal[
    "erode", "dilate", "open", "close", "gradient", "bottom_hat", "top_hat"
]
MORPHOLOGY_OPERATIONS = frozenset(get_args(Morphology_Operations))


class Morphology(BaseNode):
    __slots__ = ("image", "operation", "kernel_size")
    CATEGORY = "image/filters"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"image": "IMAGE", "operation": "COMBO", "kernel_size": "INT"}

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        operation: Morphology_Operations | tuple[str, int],
        kernel_size: int | tuple[str, int] = 3,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.operation = operation
        self.kernel_size = kernel_size
        self.__validate()

    def __validate(self) -> None:
        validate_choice("operation", self.operation, MORPHOLOGY_OPERATIONS)
        validate_range("kernel_size", self.kernel_size, 3, 999)


ColorTransfer_Methods = Literal["reinhard_lab", "mkl_lab", "histogram"]


COLORTRANSFER_METHODS = frozenset(get_args(ColorTransfer_Methods))


ColorTransfer_Source_Stats = Literal["per_frame", "uniform", "target_frame"]


class ColorTransfer(DynamicInputs, BaseNode):
    __slots__ = (
        "image_target",
        "image_ref",
        "method",
        "source_stats",
        "source_stats_target_index",
        "strength",
    )
    CATEGORY = "image/filters"
    OUTPUTS = {"image": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_NAMES = {"source_stats_target_index": "source_stats.target_index"}
    INPUT_SOCKET_TYPES = {
        "image_target": "IMAGE",
        "image_ref": "IMAGE",
        "method": "COMBO",
        "source_stats": "COMFY_DYNAMICCOMBO_V3",
        "source_stats_target_index": "INT",
        "strength": "FLOAT",
    }
    DYNAMIC_INPUTS = {
        "source_stats": Combo(
            {
                "per_frame": {},
                "uniform": {},
                "target_frame": {
                    "source_stats_target_index": Number(default=0, low=0, high=10000)
                },
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        image_target: tuple[str, int],
        image_ref: tuple[str, int],
        method: ColorTransfer_Methods | tuple[str, int],
        source_stats: ColorTransfer_Source_Stats = "per_frame",
        source_stats_target_index: int | tuple[str, int] | None = None,
        strength: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image_target = image_target
        self.image_ref = image_ref
        self.method = method
        self.source_stats = source_stats
        if source_stats_target_index is not None:
            self.source_stats_target_index = source_stats_target_index
        self.strength = strength
        self.__validate()

    def __validate(self) -> None:
        validate_choice("method", self.method, COLORTRANSFER_METHODS)
        validate_range("strength", self.strength, 0.0, 10.0)
        validate_dynamic(self)
