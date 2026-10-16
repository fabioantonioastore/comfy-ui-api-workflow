from typing import Literal, get_args

from ..base import BaseNode
from ..validation import validate_bool, validate_choice, validate_range


class MaskToImage(BaseNode):
    __slots__ = ("mask",)
    CATEGORY = "image/mask"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"mask": "MASK"}

    def __init__(self, node_id: str, mask: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.mask = mask


ImageToMask_Channels = Literal["red", "green", "blue", "alpha"]
IMAGETOMASK_CHANNELS = frozenset(get_args(ImageToMask_Channels))


class ImageToMask(BaseNode):
    __slots__ = ("image", "channel")
    CATEGORY = "image/mask"
    OUTPUTS = {"MASK": 0}
    RETURN_TYPES = ("MASK",)
    INPUT_SOCKET_TYPES = {"image": "IMAGE", "channel": "COMBO"}

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        channel: ImageToMask_Channels | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.channel = channel
        self.__validate()

    def __validate(self) -> None:
        validate_choice("channel", self.channel, IMAGETOMASK_CHANNELS)


class ImageColorToMask(BaseNode):
    __slots__ = ("image", "color")
    CATEGORY = "image/mask"
    OUTPUTS = {"MASK": 0}
    RETURN_TYPES = ("MASK",)
    INPUT_SOCKET_TYPES = {"image": "IMAGE", "color": "INT"}

    def __init__(
        self, node_id: str, image: tuple[str, int], color: int | tuple[str, int] = 0
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.color = color
        self.__validate()

    def __validate(self) -> None:
        validate_range("color", self.color, 0, 16_777_215)


class SolidMask(BaseNode):
    __slots__ = ("value", "width", "height")
    CATEGORY = "image/mask"
    OUTPUTS = {"MASK": 0}
    RETURN_TYPES = ("MASK",)
    INPUT_SOCKET_TYPES = {"value": "FLOAT", "width": "INT", "height": "INT"}

    def __init__(
        self,
        node_id: str,
        value: float | tuple[str, int] = 1.0,
        width: int | tuple[str, int] = 512,
        height: int | tuple[str, int] = 512,
    ) -> None:
        super().__init__(node_id=node_id)
        self.value = value
        self.width = width
        self.height = height
        self.__validate()

    def __validate(self) -> None:
        validate_range("value", self.value, 0.0, 1.0)
        validate_range("width", self.width, 1, 16_384)
        validate_range("height", self.height, 1, 16_384)


class InvertMask(BaseNode):
    __slots__ = ("mask",)
    CATEGORY = "image/mask"
    OUTPUTS = {"MASK": 0}
    RETURN_TYPES = ("MASK",)
    INPUT_SOCKET_TYPES = {"mask": "MASK"}

    def __init__(self, node_id: str, mask: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.mask = mask


class CropMask(BaseNode):
    __slots__ = ("mask", "x", "y", "width", "height")
    CATEGORY = "image/mask"
    OUTPUTS = {"MASK": 0}
    RETURN_TYPES = ("MASK",)
    INPUT_SOCKET_TYPES = {
        "mask": "MASK",
        "x": "INT",
        "y": "INT",
        "width": "INT",
        "height": "INT",
    }

    def __init__(
        self,
        node_id: str,
        mask: tuple[str, int],
        x: int | tuple[str, int] = 0,
        y: int | tuple[str, int] = 0,
        width: int | tuple[str, int] = 512,
        height: int | tuple[str, int] = 512,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mask = mask
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.__validate()

    def __validate(self) -> None:
        validate_range("x", self.x, 0, 16_384)
        validate_range("y", self.y, 0, 16_384)
        validate_range("width", self.width, 1, 16_384)
        validate_range("height", self.height, 1, 16_384)


MaskComposite_Operations = Literal["multiply", "add", "subtract", "and", "or", "xor"]
MASKCOMPOSITE_OPERATIONS = frozenset(get_args(MaskComposite_Operations))


class MaskComposite(BaseNode):
    __slots__ = ("destination", "source", "operation", "x", "y")
    CATEGORY = "image/mask"
    OUTPUTS = {"MASK": 0}
    RETURN_TYPES = ("MASK",)
    INPUT_SOCKET_TYPES = {
        "destination": "MASK",
        "source": "MASK",
        "x": "INT",
        "y": "INT",
        "operation": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        destination: tuple[str, int],
        source: tuple[str, int],
        operation: MaskComposite_Operations | tuple[str, int],
        x: int | tuple[str, int] = 0,
        y: int | tuple[str, int] = 0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.destination = destination
        self.source = source
        self.operation = operation
        self.x = x
        self.y = y
        self.__validate()

    def __validate(self) -> None:
        validate_choice("operation", self.operation, MASKCOMPOSITE_OPERATIONS)
        validate_range("x", self.x, 0, 16_384)
        validate_range("y", self.y, 0, 16_384)


class FeatherMask(BaseNode):
    __slots__ = ("mask", "left", "top", "right", "bottom")
    CATEGORY = "image/mask"
    OUTPUTS = {"MASK": 0}
    RETURN_TYPES = ("MASK",)
    INPUT_SOCKET_TYPES = {
        "mask": "MASK",
        "left": "INT",
        "top": "INT",
        "right": "INT",
        "bottom": "INT",
    }

    def __init__(
        self,
        node_id: str,
        mask: tuple[str, int],
        left: int | tuple[str, int] = 0,
        top: int | tuple[str, int] = 0,
        right: int | tuple[str, int] = 0,
        bottom: int | tuple[str, int] = 0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mask = mask
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.__validate()

    def __validate(self) -> None:
        validate_range("left", self.left, 0, 16_384)
        validate_range("top", self.top, 0, 16_384)
        validate_range("right", self.right, 0, 16_384)
        validate_range("bottom", self.bottom, 0, 16_384)


class GrowMask(BaseNode):
    __slots__ = ("mask", "expand", "tapered_corners")
    CATEGORY = "image/mask"
    OUTPUTS = {"MASK": 0}
    RETURN_TYPES = ("MASK",)
    INPUT_SOCKET_TYPES = {"mask": "MASK", "expand": "INT", "tapered_corners": "BOOLEAN"}

    def __init__(
        self,
        node_id: str,
        mask: tuple[str, int],
        expand: int | tuple[str, int] = 0,
        tapered_corners: bool | tuple[str, int] = True,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mask = mask
        self.expand = expand
        self.tapered_corners = tapered_corners
        self.__validate()

    def __validate(self) -> None:
        validate_range("expand", self.expand, -16_384, 16_384)
        validate_bool("tapered_corners", self.tapered_corners)


class ThresholdMask(BaseNode):
    __slots__ = ("mask", "value")
    CATEGORY = "image/mask"
    OUTPUTS = {"MASK": 0}
    RETURN_TYPES = ("MASK",)
    INPUT_SOCKET_TYPES = {"mask": "MASK", "value": "FLOAT"}

    def __init__(
        self, node_id: str, mask: tuple[str, int], value: float | tuple[str, int] = 0.5
    ) -> None:
        super().__init__(node_id=node_id)
        self.mask = mask
        self.value = value
        self.__validate()

    def __validate(self) -> None:
        validate_range("value", self.value, 0.0, 1.0)


class MaskPreview(BaseNode):
    __slots__ = ("mask",)
    CATEGORY = "image/mask"
    OUTPUTS = {"mask": 0}
    RETURN_TYPES = ("MASK",)
    INPUT_SOCKET_TYPES = {"mask": "MASK"}

    def __init__(self, node_id: str, mask: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.mask = mask


class VOIDQuadmaskPreprocess(BaseNode):
    __slots__ = ("mask", "dilate_width")
    CATEGORY = "image/mask"
    OUTPUTS = {"quadmask": 0}
    RETURN_TYPES = ("MASK",)
    INPUT_SOCKET_TYPES = {"mask": "MASK", "dilate_width": "INT"}

    def __init__(
        self,
        node_id: str,
        mask: tuple[str, int],
        dilate_width: int | tuple[str, int] = 0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mask = mask
        self.dilate_width = dilate_width
        self.__validate()

    def __validate(self) -> None:
        validate_range("dilate_width", self.dilate_width, 0, 50)
