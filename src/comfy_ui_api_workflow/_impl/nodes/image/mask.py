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


class BatchMasksNode(BaseNode):
    __slots__ = (
        "masks_mask0",
        "masks_mask1",
        "masks_mask2",
        "masks_mask3",
        "masks_mask4",
        "masks_mask5",
        "masks_mask6",
        "masks_mask7",
        "masks_mask8",
        "masks_mask9",
        "masks_mask10",
        "masks_mask11",
        "masks_mask12",
        "masks_mask13",
        "masks_mask14",
        "masks_mask15",
        "masks_mask16",
        "masks_mask17",
        "masks_mask18",
        "masks_mask19",
        "masks_mask20",
        "masks_mask21",
        "masks_mask22",
        "masks_mask23",
        "masks_mask24",
        "masks_mask25",
        "masks_mask26",
        "masks_mask27",
        "masks_mask28",
        "masks_mask29",
        "masks_mask30",
        "masks_mask31",
        "masks_mask32",
        "masks_mask33",
        "masks_mask34",
        "masks_mask35",
        "masks_mask36",
        "masks_mask37",
        "masks_mask38",
        "masks_mask39",
        "masks_mask40",
        "masks_mask41",
        "masks_mask42",
        "masks_mask43",
        "masks_mask44",
        "masks_mask45",
        "masks_mask46",
        "masks_mask47",
        "masks_mask48",
        "masks_mask49",
    )
    CATEGORY = "image/mask"
    OUTPUTS = {"MASK": 0}
    RETURN_TYPES = ("MASK",)
    INPUT_NAMES = {
        "masks_mask0": "masks.mask0",
        "masks_mask1": "masks.mask1",
        "masks_mask2": "masks.mask2",
        "masks_mask3": "masks.mask3",
        "masks_mask4": "masks.mask4",
        "masks_mask5": "masks.mask5",
        "masks_mask6": "masks.mask6",
        "masks_mask7": "masks.mask7",
        "masks_mask8": "masks.mask8",
        "masks_mask9": "masks.mask9",
        "masks_mask10": "masks.mask10",
        "masks_mask11": "masks.mask11",
        "masks_mask12": "masks.mask12",
        "masks_mask13": "masks.mask13",
        "masks_mask14": "masks.mask14",
        "masks_mask15": "masks.mask15",
        "masks_mask16": "masks.mask16",
        "masks_mask17": "masks.mask17",
        "masks_mask18": "masks.mask18",
        "masks_mask19": "masks.mask19",
        "masks_mask20": "masks.mask20",
        "masks_mask21": "masks.mask21",
        "masks_mask22": "masks.mask22",
        "masks_mask23": "masks.mask23",
        "masks_mask24": "masks.mask24",
        "masks_mask25": "masks.mask25",
        "masks_mask26": "masks.mask26",
        "masks_mask27": "masks.mask27",
        "masks_mask28": "masks.mask28",
        "masks_mask29": "masks.mask29",
        "masks_mask30": "masks.mask30",
        "masks_mask31": "masks.mask31",
        "masks_mask32": "masks.mask32",
        "masks_mask33": "masks.mask33",
        "masks_mask34": "masks.mask34",
        "masks_mask35": "masks.mask35",
        "masks_mask36": "masks.mask36",
        "masks_mask37": "masks.mask37",
        "masks_mask38": "masks.mask38",
        "masks_mask39": "masks.mask39",
        "masks_mask40": "masks.mask40",
        "masks_mask41": "masks.mask41",
        "masks_mask42": "masks.mask42",
        "masks_mask43": "masks.mask43",
        "masks_mask44": "masks.mask44",
        "masks_mask45": "masks.mask45",
        "masks_mask46": "masks.mask46",
        "masks_mask47": "masks.mask47",
        "masks_mask48": "masks.mask48",
        "masks_mask49": "masks.mask49",
    }
    INPUT_SOCKET_TYPES = {
        "masks_mask0": "MASK",
        "masks_mask1": "MASK",
        "masks_mask2": "MASK",
        "masks_mask3": "MASK",
        "masks_mask4": "MASK",
        "masks_mask5": "MASK",
        "masks_mask6": "MASK",
        "masks_mask7": "MASK",
        "masks_mask8": "MASK",
        "masks_mask9": "MASK",
        "masks_mask10": "MASK",
        "masks_mask11": "MASK",
        "masks_mask12": "MASK",
        "masks_mask13": "MASK",
        "masks_mask14": "MASK",
        "masks_mask15": "MASK",
        "masks_mask16": "MASK",
        "masks_mask17": "MASK",
        "masks_mask18": "MASK",
        "masks_mask19": "MASK",
        "masks_mask20": "MASK",
        "masks_mask21": "MASK",
        "masks_mask22": "MASK",
        "masks_mask23": "MASK",
        "masks_mask24": "MASK",
        "masks_mask25": "MASK",
        "masks_mask26": "MASK",
        "masks_mask27": "MASK",
        "masks_mask28": "MASK",
        "masks_mask29": "MASK",
        "masks_mask30": "MASK",
        "masks_mask31": "MASK",
        "masks_mask32": "MASK",
        "masks_mask33": "MASK",
        "masks_mask34": "MASK",
        "masks_mask35": "MASK",
        "masks_mask36": "MASK",
        "masks_mask37": "MASK",
        "masks_mask38": "MASK",
        "masks_mask39": "MASK",
        "masks_mask40": "MASK",
        "masks_mask41": "MASK",
        "masks_mask42": "MASK",
        "masks_mask43": "MASK",
        "masks_mask44": "MASK",
        "masks_mask45": "MASK",
        "masks_mask46": "MASK",
        "masks_mask47": "MASK",
        "masks_mask48": "MASK",
        "masks_mask49": "MASK",
    }

    def __init__(
        self,
        node_id: str,
        masks_mask0: tuple[str, int],
        masks_mask1: tuple[str, int] | None = None,
        masks_mask2: tuple[str, int] | None = None,
        masks_mask3: tuple[str, int] | None = None,
        masks_mask4: tuple[str, int] | None = None,
        masks_mask5: tuple[str, int] | None = None,
        masks_mask6: tuple[str, int] | None = None,
        masks_mask7: tuple[str, int] | None = None,
        masks_mask8: tuple[str, int] | None = None,
        masks_mask9: tuple[str, int] | None = None,
        masks_mask10: tuple[str, int] | None = None,
        masks_mask11: tuple[str, int] | None = None,
        masks_mask12: tuple[str, int] | None = None,
        masks_mask13: tuple[str, int] | None = None,
        masks_mask14: tuple[str, int] | None = None,
        masks_mask15: tuple[str, int] | None = None,
        masks_mask16: tuple[str, int] | None = None,
        masks_mask17: tuple[str, int] | None = None,
        masks_mask18: tuple[str, int] | None = None,
        masks_mask19: tuple[str, int] | None = None,
        masks_mask20: tuple[str, int] | None = None,
        masks_mask21: tuple[str, int] | None = None,
        masks_mask22: tuple[str, int] | None = None,
        masks_mask23: tuple[str, int] | None = None,
        masks_mask24: tuple[str, int] | None = None,
        masks_mask25: tuple[str, int] | None = None,
        masks_mask26: tuple[str, int] | None = None,
        masks_mask27: tuple[str, int] | None = None,
        masks_mask28: tuple[str, int] | None = None,
        masks_mask29: tuple[str, int] | None = None,
        masks_mask30: tuple[str, int] | None = None,
        masks_mask31: tuple[str, int] | None = None,
        masks_mask32: tuple[str, int] | None = None,
        masks_mask33: tuple[str, int] | None = None,
        masks_mask34: tuple[str, int] | None = None,
        masks_mask35: tuple[str, int] | None = None,
        masks_mask36: tuple[str, int] | None = None,
        masks_mask37: tuple[str, int] | None = None,
        masks_mask38: tuple[str, int] | None = None,
        masks_mask39: tuple[str, int] | None = None,
        masks_mask40: tuple[str, int] | None = None,
        masks_mask41: tuple[str, int] | None = None,
        masks_mask42: tuple[str, int] | None = None,
        masks_mask43: tuple[str, int] | None = None,
        masks_mask44: tuple[str, int] | None = None,
        masks_mask45: tuple[str, int] | None = None,
        masks_mask46: tuple[str, int] | None = None,
        masks_mask47: tuple[str, int] | None = None,
        masks_mask48: tuple[str, int] | None = None,
        masks_mask49: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.masks_mask0 = masks_mask0
        if masks_mask1 is not None:
            self.masks_mask1 = masks_mask1
        if masks_mask2 is not None:
            self.masks_mask2 = masks_mask2
        if masks_mask3 is not None:
            self.masks_mask3 = masks_mask3
        if masks_mask4 is not None:
            self.masks_mask4 = masks_mask4
        if masks_mask5 is not None:
            self.masks_mask5 = masks_mask5
        if masks_mask6 is not None:
            self.masks_mask6 = masks_mask6
        if masks_mask7 is not None:
            self.masks_mask7 = masks_mask7
        if masks_mask8 is not None:
            self.masks_mask8 = masks_mask8
        if masks_mask9 is not None:
            self.masks_mask9 = masks_mask9
        if masks_mask10 is not None:
            self.masks_mask10 = masks_mask10
        if masks_mask11 is not None:
            self.masks_mask11 = masks_mask11
        if masks_mask12 is not None:
            self.masks_mask12 = masks_mask12
        if masks_mask13 is not None:
            self.masks_mask13 = masks_mask13
        if masks_mask14 is not None:
            self.masks_mask14 = masks_mask14
        if masks_mask15 is not None:
            self.masks_mask15 = masks_mask15
        if masks_mask16 is not None:
            self.masks_mask16 = masks_mask16
        if masks_mask17 is not None:
            self.masks_mask17 = masks_mask17
        if masks_mask18 is not None:
            self.masks_mask18 = masks_mask18
        if masks_mask19 is not None:
            self.masks_mask19 = masks_mask19
        if masks_mask20 is not None:
            self.masks_mask20 = masks_mask20
        if masks_mask21 is not None:
            self.masks_mask21 = masks_mask21
        if masks_mask22 is not None:
            self.masks_mask22 = masks_mask22
        if masks_mask23 is not None:
            self.masks_mask23 = masks_mask23
        if masks_mask24 is not None:
            self.masks_mask24 = masks_mask24
        if masks_mask25 is not None:
            self.masks_mask25 = masks_mask25
        if masks_mask26 is not None:
            self.masks_mask26 = masks_mask26
        if masks_mask27 is not None:
            self.masks_mask27 = masks_mask27
        if masks_mask28 is not None:
            self.masks_mask28 = masks_mask28
        if masks_mask29 is not None:
            self.masks_mask29 = masks_mask29
        if masks_mask30 is not None:
            self.masks_mask30 = masks_mask30
        if masks_mask31 is not None:
            self.masks_mask31 = masks_mask31
        if masks_mask32 is not None:
            self.masks_mask32 = masks_mask32
        if masks_mask33 is not None:
            self.masks_mask33 = masks_mask33
        if masks_mask34 is not None:
            self.masks_mask34 = masks_mask34
        if masks_mask35 is not None:
            self.masks_mask35 = masks_mask35
        if masks_mask36 is not None:
            self.masks_mask36 = masks_mask36
        if masks_mask37 is not None:
            self.masks_mask37 = masks_mask37
        if masks_mask38 is not None:
            self.masks_mask38 = masks_mask38
        if masks_mask39 is not None:
            self.masks_mask39 = masks_mask39
        if masks_mask40 is not None:
            self.masks_mask40 = masks_mask40
        if masks_mask41 is not None:
            self.masks_mask41 = masks_mask41
        if masks_mask42 is not None:
            self.masks_mask42 = masks_mask42
        if masks_mask43 is not None:
            self.masks_mask43 = masks_mask43
        if masks_mask44 is not None:
            self.masks_mask44 = masks_mask44
        if masks_mask45 is not None:
            self.masks_mask45 = masks_mask45
        if masks_mask46 is not None:
            self.masks_mask46 = masks_mask46
        if masks_mask47 is not None:
            self.masks_mask47 = masks_mask47
        if masks_mask48 is not None:
            self.masks_mask48 = masks_mask48
        if masks_mask49 is not None:
            self.masks_mask49 = masks_mask49
