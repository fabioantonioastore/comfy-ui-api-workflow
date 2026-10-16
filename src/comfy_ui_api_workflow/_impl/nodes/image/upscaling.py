from typing import Literal, get_args

from ..base import BaseNode
from ..validation import validate_choice, validate_range

ImageScale_Upscale_Methods = Literal[
    "nearest-exact", "bilinear", "area", "bicubic", "lanczos"
]
ImageScale_Crops = Literal["disabled", "center"]
IMAGESCALE_UPSCALE_METHODS = frozenset(get_args(ImageScale_Upscale_Methods))
IMAGESCALE_CROPS = frozenset(get_args(ImageScale_Crops))


class ImageScale(BaseNode):
    __slots__ = ("image", "upscale_method", "crop", "width", "height")
    CATEGORY = "image/upscaling"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "image": "IMAGE",
        "upscale_method": "V1 COMBO",
        "width": "INT",
        "height": "INT",
        "crop": "V1 COMBO",
    }

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        upscale_method: ImageScale_Upscale_Methods | tuple[str, int],
        crop: ImageScale_Crops | tuple[str, int],
        width: int | tuple[str, int] = 512,
        height: int | tuple[str, int] = 512,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.upscale_method = upscale_method
        self.crop = crop
        self.width = width
        self.height = height
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "upscale_method", self.upscale_method, IMAGESCALE_UPSCALE_METHODS
        )
        validate_choice("crop", self.crop, IMAGESCALE_CROPS)
        validate_range("width", self.width, 0, 16_384)
        validate_range("height", self.height, 0, 16_384)


ImageScaleBy_Upscale_Methods = Literal[
    "nearest-exact", "bilinear", "area", "bicubic", "lanczos"
]
IMAGESCALEBY_UPSCALE_METHODS = frozenset(get_args(ImageScaleBy_Upscale_Methods))


class ImageScaleBy(BaseNode):
    __slots__ = ("image", "upscale_method", "scale_by")
    CATEGORY = "image/upscaling"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "image": "IMAGE",
        "upscale_method": "V1 COMBO",
        "scale_by": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        upscale_method: ImageScaleBy_Upscale_Methods | tuple[str, int],
        scale_by: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.upscale_method = upscale_method
        self.scale_by = scale_by
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "upscale_method", self.upscale_method, IMAGESCALEBY_UPSCALE_METHODS
        )
        validate_range("scale_by", self.scale_by, 0.01, 8.0)


class ImageUpscaleWithModel(BaseNode):
    __slots__ = ("upscale_model", "image")
    CATEGORY = "image/upscaling"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"upscale_model": "UPSCALE_MODEL", "image": "IMAGE"}

    def __init__(
        self, node_id: str, upscale_model: tuple[str, int], image: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.upscale_model = upscale_model
        self.image = image


ImageScaleToTotalPixels_Upscale_Methods = Literal[
    "nearest-exact", "bilinear", "area", "bicubic", "lanczos"
]
IMAGESCALETOTOTALPIXELS_UPSCALE_METHODS = frozenset(
    get_args(ImageScaleToTotalPixels_Upscale_Methods)
)


class ImageScaleToTotalPixels(BaseNode):
    __slots__ = ("image", "upscale_method", "megapixels", "resolution_steps")
    CATEGORY = "image/upscaling"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "image": "IMAGE",
        "upscale_method": "COMBO",
        "megapixels": "FLOAT",
        "resolution_steps": "INT",
    }

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        upscale_method: ImageScaleToTotalPixels_Upscale_Methods | tuple[str, int],
        megapixels: float | tuple[str, int] = 1.0,
        resolution_steps: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.upscale_method = upscale_method
        self.megapixels = megapixels
        self.resolution_steps = resolution_steps
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "upscale_method",
            self.upscale_method,
            IMAGESCALETOTOTALPIXELS_UPSCALE_METHODS,
        )
        validate_range("megapixels", self.megapixels, 0.01, 16.0)
        validate_range("resolution_steps", self.resolution_steps, 1, 256)


ImageScaleToMaxDimension_Upscale_Methods = Literal[
    "area", "lanczos", "bilinear", "nearest-exact", "bilinear", "bicubic"
]
IMAGESCALETOMAXDIMENSION_UPSCALE_METHODS = frozenset(
    get_args(ImageScaleToMaxDimension_Upscale_Methods)
)


class ImageScaleToMaxDimension(BaseNode):
    __slots__ = ("image", "upscale_method", "largest_size")
    CATEGORY = "image/upscaling"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "image": "IMAGE",
        "upscale_method": "COMBO",
        "largest_size": "INT",
    }

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        upscale_method: ImageScaleToMaxDimension_Upscale_Methods | tuple[str, int],
        largest_size: int | tuple[str, int] = 512,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.upscale_method = upscale_method
        self.largest_size = largest_size
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "upscale_method",
            self.upscale_method,
            IMAGESCALETOMAXDIMENSION_UPSCALE_METHODS,
        )
        validate_range("largest_size", self.largest_size, 0, 16_384)
