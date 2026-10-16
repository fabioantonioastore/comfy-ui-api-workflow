from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice, validate_range

LatentRotate_Rotations = Literal["none", "90 degrees", "180 degrees", "270 degrees"]
LATENTROTATE_ROTATIONS = frozenset(get_args(LatentRotate_Rotations))


class LatentRotate(BaseNode):
    __slots__ = ("samples", "rotation")
    CATEGORY = "model/latent/transform"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"samples": "LATENT", "rotation": "V1 COMBO"}

    def __init__(
        self,
        node_id: str,
        samples: tuple[str, int],
        rotation: LatentRotate_Rotations | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.rotation = rotation
        self.__validate()

    def __validate(self) -> None:
        validate_choice("rotation", self.rotation, LATENTROTATE_ROTATIONS)


LatentFlip_Flip_Methods = Literal["x-axis: vertically", "y-axis: horizontally"]
LATENTFLIP_FLIP_METHODS = frozenset(get_args(LatentFlip_Flip_Methods))


class LatentFlip(BaseNode):
    __slots__ = ("samples", "flip_method")
    CATEGORY = "model/latent/transform"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"samples": "LATENT", "flip_method": "V1 COMBO"}

    def __init__(
        self,
        node_id: str,
        samples: tuple[str, int],
        flip_method: LatentFlip_Flip_Methods | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.flip_method = flip_method
        self.__validate()

    def __validate(self) -> None:
        validate_choice("flip_method", self.flip_method, LATENTFLIP_FLIP_METHODS)


class LatentCrop(BaseNode):
    __slots__ = ("samples", "width", "height", "x", "y")
    CATEGORY = "model/latent/transform"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "samples": "LATENT",
        "width": "INT",
        "height": "INT",
        "x": "INT",
        "y": "INT",
    }

    def __init__(
        self,
        node_id: str,
        samples: tuple[str, int],
        width: int | tuple[str, int] = 512,
        height: int | tuple[str, int] = 512,
        x: int | tuple[str, int] = 0,
        y: int | tuple[str, int] = 0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 64, 16_384)
        validate_range("height", self.height, 64, 16_384)
        validate_range("x", self.x, 0, 16_384)
        validate_range("y", self.y, 0, 16_384)
