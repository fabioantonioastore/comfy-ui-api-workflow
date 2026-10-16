from typing import Literal, get_args

from ..base import BaseNode
from ..validation import validate_bool, validate_choice, validate_range


class ImageCompositeMasked(BaseNode):
    __slots__ = ("destination", "source", "x", "y", "resize_source", "mask")
    CATEGORY = "image/compositing"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "destination": "IMAGE",
        "source": "IMAGE",
        "x": "INT",
        "y": "INT",
        "resize_source": "BOOLEAN",
        "mask": "MASK",
    }

    def __init__(
        self,
        node_id: str,
        destination: tuple[str, int],
        source: tuple[str, int],
        x: int | tuple[str, int] = 0,
        y: int | tuple[str, int] = 0,
        resize_source: bool | tuple[str, int] = False,
        mask: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.destination = destination
        self.source = source
        self.x = x
        self.y = y
        self.resize_source = resize_source
        if mask is not None:
            self.mask = mask
        self.__validate()

    def __validate(self) -> None:
        validate_range("x", self.x, 0, 16_384)
        validate_range("y", self.y, 0, 16_384)
        validate_bool("resize_source", self.resize_source)


PorterDuffImageComposite_Modes = Literal[
    "ADD",
    "CLEAR",
    "DARKEN",
    "DST",
    "DST_ATOP",
    "DST_IN",
    "DST_OUT",
    "DST_OVER",
    "LIGHTEN",
    "MULTIPLY",
    "OVERLAY",
    "SCREEN",
    "SRC",
    "SRC_ATOP",
    "SRC_IN",
    "SRC_OUT",
    "SRC_OVER",
    "XOR",
]
PORTERDUFFIMAGECOMPOSITE_MODES = frozenset(get_args(PorterDuffImageComposite_Modes))


class PorterDuffImageComposite(BaseNode):
    __slots__ = ("source", "source_alpha", "destination", "destination_alpha", "mode")
    CATEGORY = "image/compositing"
    OUTPUTS = {"IMAGE": 0, "MASK": 1}
    RETURN_TYPES = ("IMAGE", "MASK")
    INPUT_SOCKET_TYPES = {
        "source": "IMAGE",
        "source_alpha": "MASK",
        "destination": "IMAGE",
        "destination_alpha": "MASK",
        "mode": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        source: tuple[str, int],
        source_alpha: tuple[str, int],
        destination: tuple[str, int],
        destination_alpha: tuple[str, int],
        mode: PorterDuffImageComposite_Modes | tuple[str, int] = "DST",
    ) -> None:
        super().__init__(node_id=node_id)
        self.source = source
        self.source_alpha = source_alpha
        self.destination = destination
        self.destination_alpha = destination_alpha
        self.mode = mode
        self.__validate()

    def __validate(self) -> None:
        validate_choice("mode", self.mode, PORTERDUFFIMAGECOMPOSITE_MODES)


class SplitImageWithAlpha(BaseNode):
    __slots__ = ("image",)
    CATEGORY = "image/compositing"
    OUTPUTS = {"IMAGE": 0, "MASK": 1}
    RETURN_TYPES = ("IMAGE", "MASK")
    INPUT_SOCKET_TYPES = {"image": "IMAGE"}

    def __init__(self, node_id: str, image: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.image = image


class JoinImageWithAlpha(BaseNode):
    __slots__ = ("image", "alpha")
    CATEGORY = "image/compositing"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"image": "IMAGE", "alpha": "MASK"}

    def __init__(
        self, node_id: str, image: tuple[str, int], alpha: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.alpha = alpha
