from typing import Literal, get_args

from ..base import BaseNode
from ..validation import validate_choice, validate_range


class ImageInvert(BaseNode):
    __slots__ = ("image",)
    CATEGORY = "image/color"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"image": "IMAGE"}

    def __init__(self, node_id: str, image: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.image = image


ImageColorSpace_Sources = Literal[
    "sRGB", "HDR", "HDR PQ", "linear", "HDR LogC3", "HDR ACEScct"
]
ImageColorSpace_Destinations = Literal[
    "sRGB", "HDR", "HDR PQ", "linear", "HDR LogC3", "HDR ACEScct"
]
IMAGECOLORSPACE_SOURCES = frozenset(get_args(ImageColorSpace_Sources))
IMAGECOLORSPACE_DESTINATIONS = frozenset(get_args(ImageColorSpace_Destinations))


class ImageColorSpace(BaseNode):
    __slots__ = ("image", "source", "destination")
    CATEGORY = "image/color"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"image": "IMAGE", "source": "COMBO", "destination": "COMBO"}

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        source: ImageColorSpace_Sources | tuple[str, int] = "sRGB",
        destination: ImageColorSpace_Destinations | tuple[str, int] = "sRGB",
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.source = source
        self.destination = destination
        self.__validate()

    def __validate(self) -> None:
        validate_choice("source", self.source, IMAGECOLORSPACE_SOURCES)
        validate_choice("destination", self.destination, IMAGECOLORSPACE_DESTINATIONS)


class NormalizeImages(BaseNode):
    __slots__ = ("images", "mean", "std")
    CATEGORY = "image/color"
    OUTPUTS = {"images": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"images": "IMAGE", "mean": "FLOAT", "std": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        mean: float | tuple[str, int] = 0.5,
        std: float | tuple[str, int] = 0.5,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.mean = mean
        self.std = std
        self.__validate()

    def __validate(self) -> None:
        validate_range("mean", self.mean, 0.0, 1.0)
        validate_range("std", self.std, 0.001, 1.0)


class ImageRGBToYUV(BaseNode):
    __slots__ = ("image",)
    CATEGORY = "image/color"
    OUTPUTS = {"Y": 0, "U": 1, "V": 2}
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    INPUT_SOCKET_TYPES = {"image": "IMAGE"}

    def __init__(self, node_id: str, image: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.image = image


class ImageYUVToRGB(BaseNode):
    __slots__ = ("Y", "U", "V")
    CATEGORY = "image/color"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"Y": "IMAGE", "U": "IMAGE", "V": "IMAGE"}

    def __init__(
        self, node_id: str, Y: tuple[str, int], U: tuple[str, int], V: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.Y = Y
        self.U = U
        self.V = V
