from typing import Any, Literal, get_args

from ..base import BaseNode
from ..validation import validate_bool, validate_choice, validate_range
from ..dynamic import Choice, Combo, DynamicInputs, Link, Number, validate_dynamic


class ImagePadForOutpaint(BaseNode):
    __slots__ = ("image", "left", "top", "right", "bottom", "feathering")
    CATEGORY = "image/transform"
    OUTPUTS = {"IMAGE": 0, "MASK": 1}
    RETURN_TYPES = ("IMAGE", "MASK")
    INPUT_SOCKET_TYPES = {
        "image": "IMAGE",
        "left": "INT",
        "top": "INT",
        "right": "INT",
        "bottom": "INT",
        "feathering": "INT",
    }

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        left: int | tuple[str, int] = 0,
        top: int | tuple[str, int] = 0,
        right: int | tuple[str, int] = 0,
        bottom: int | tuple[str, int] = 0,
        feathering: int | tuple[str, int] = 40,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.feathering = feathering
        self.__validate()

    def __validate(self) -> None:
        validate_range("left", self.left, 0, 16_384)
        validate_range("top", self.top, 0, 16_384)
        validate_range("right", self.right, 0, 16_384)
        validate_range("bottom", self.bottom, 0, 16_384)
        validate_range("feathering", self.feathering, 0, 16_384)


class ImageCropV2(BaseNode):
    __slots__ = ("image", "crop_region")
    CATEGORY = "image/transform"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"image": "IMAGE", "crop_region": "BOUNDING_BOX"}

    def __init__(
        self, node_id: str, image: tuple[str, int], crop_region: Any = None
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.crop_region = (
            {"x": 0, "y": 0, "width": 512, "height": 512}
            if crop_region is None
            else crop_region
        )


class ImageCropToMask(BaseNode):
    __slots__ = (
        "images",
        "masks",
        "width",
        "height",
        "pad_factor",
        "grow_mask",
        "background",
    )
    CATEGORY = "image/transform"
    OUTPUTS = {"images": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "images": "IMAGE",
        "masks": "MASK",
        "width": "INT",
        "height": "INT",
        "pad_factor": "FLOAT",
        "grow_mask": "INT",
        "background": "COLOR",
    }

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        masks: tuple[str, int],
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
        pad_factor: float | tuple[str, int] = 1.0,
        grow_mask: int | tuple[str, int] = 0,
        background: Any = "#000000",
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.masks = masks
        self.width = width
        self.height = height
        self.pad_factor = pad_factor
        self.grow_mask = grow_mask
        self.background = background
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 64, 4096)
        validate_range("height", self.height, 64, 4096)
        validate_range("pad_factor", self.pad_factor, 1.0, 2.0)
        validate_range("grow_mask", self.grow_mask, -32, 32)


ImageStitch_Directions = Literal["right", "down", "left", "up"]
ImageStitch_Spacing_Colors = Literal["white", "black", "red", "green", "blue"]
IMAGESTITCH_DIRECTIONS = frozenset(get_args(ImageStitch_Directions))
IMAGESTITCH_SPACING_COLORS = frozenset(get_args(ImageStitch_Spacing_Colors))


class ImageStitch(BaseNode):
    __slots__ = (
        "image1",
        "direction",
        "match_image_size",
        "spacing_width",
        "spacing_color",
        "image2",
    )
    CATEGORY = "image/transform"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "image1": "IMAGE",
        "direction": "COMBO",
        "match_image_size": "BOOLEAN",
        "spacing_width": "INT",
        "spacing_color": "COMBO",
        "image2": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        image1: tuple[str, int],
        direction: ImageStitch_Directions | tuple[str, int] = "right",
        match_image_size: bool | tuple[str, int] = True,
        spacing_width: int | tuple[str, int] = 0,
        spacing_color: ImageStitch_Spacing_Colors | tuple[str, int] = "white",
        image2: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image1 = image1
        self.direction = direction
        self.match_image_size = match_image_size
        self.spacing_width = spacing_width
        self.spacing_color = spacing_color
        if image2 is not None:
            self.image2 = image2
        self.__validate()

    def __validate(self) -> None:
        validate_choice("direction", self.direction, IMAGESTITCH_DIRECTIONS)
        validate_bool("match_image_size", self.match_image_size)
        validate_range("spacing_width", self.spacing_width, 0, 1024)
        validate_choice("spacing_color", self.spacing_color, IMAGESTITCH_SPACING_COLORS)


ResizeAndPadImage_Padding_Colors = Literal["white", "black"]
ResizeAndPadImage_Interpolations = Literal[
    "area", "bicubic", "nearest-exact", "bilinear", "lanczos"
]
RESIZEANDPADIMAGE_PADDING_COLORS = frozenset(get_args(ResizeAndPadImage_Padding_Colors))
RESIZEANDPADIMAGE_INTERPOLATIONS = frozenset(get_args(ResizeAndPadImage_Interpolations))


class ResizeAndPadImage(BaseNode):
    __slots__ = (
        "image",
        "padding_color",
        "interpolation",
        "target_width",
        "target_height",
    )
    CATEGORY = "image/transform"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "image": "IMAGE",
        "target_width": "INT",
        "target_height": "INT",
        "padding_color": "COMBO",
        "interpolation": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        padding_color: ResizeAndPadImage_Padding_Colors | tuple[str, int],
        interpolation: ResizeAndPadImage_Interpolations | tuple[str, int],
        target_width: int | tuple[str, int] = 512,
        target_height: int | tuple[str, int] = 512,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.padding_color = padding_color
        self.interpolation = interpolation
        self.target_width = target_width
        self.target_height = target_height
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "padding_color", self.padding_color, RESIZEANDPADIMAGE_PADDING_COLORS
        )
        validate_choice(
            "interpolation", self.interpolation, RESIZEANDPADIMAGE_INTERPOLATIONS
        )
        validate_range("target_width", self.target_width, 1, 16_384)
        validate_range("target_height", self.target_height, 1, 16_384)


ImageRotate_Rotations = Literal["none", "90 degrees", "180 degrees", "270 degrees"]
IMAGEROTATE_ROTATIONS = frozenset(get_args(ImageRotate_Rotations))


class ImageRotate(BaseNode):
    __slots__ = ("image", "rotation")
    CATEGORY = "image/transform"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"image": "IMAGE", "rotation": "COMBO"}

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        rotation: ImageRotate_Rotations | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.rotation = rotation
        self.__validate()

    def __validate(self) -> None:
        validate_choice("rotation", self.rotation, IMAGEROTATE_ROTATIONS)


ImageFlip_Flip_Methods = Literal["x-axis: vertically", "y-axis: horizontally"]
IMAGEFLIP_FLIP_METHODS = frozenset(get_args(ImageFlip_Flip_Methods))


class ImageFlip(BaseNode):
    __slots__ = ("image", "flip_method")
    CATEGORY = "image/transform"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"image": "IMAGE", "flip_method": "COMBO"}

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        flip_method: ImageFlip_Flip_Methods | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.flip_method = flip_method
        self.__validate()

    def __validate(self) -> None:
        validate_choice("flip_method", self.flip_method, IMAGEFLIP_FLIP_METHODS)


class CenterCropImages(BaseNode):
    __slots__ = ("images", "width", "height")
    CATEGORY = "image/transform"
    OUTPUTS = {"images": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"images": "IMAGE", "width": "INT", "height": "INT"}

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        width: int | tuple[str, int] = 512,
        height: int | tuple[str, int] = 512,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.width = width
        self.height = height
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 1, 8192)
        validate_range("height", self.height, 1, 8192)


class RandomCropImages(BaseNode):
    __slots__ = ("images", "width", "height", "seed")
    CATEGORY = "image/transform"
    OUTPUTS = {"images": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "images": "IMAGE",
        "width": "INT",
        "height": "INT",
        "seed": "INT",
    }

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        width: int | tuple[str, int] = 512,
        height: int | tuple[str, int] = 512,
        seed: int | tuple[str, int] = 0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.width = width
        self.height = height
        self.seed = seed
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 1, 8192)
        validate_range("height", self.height, 1, 8192)
        validate_range("seed", self.seed, 0, 18_446_744_073_709_551_615)


CropByBBoxes_Keep_Aspects = Literal["stretch", "pad"]
CROPBYBBOXES_KEEP_ASPECTS = frozenset(get_args(CropByBBoxes_Keep_Aspects))


class CropByBBoxes(BaseNode):
    __slots__ = (
        "image",
        "bboxes",
        "output_width",
        "output_height",
        "padding",
        "keep_aspect",
    )
    CATEGORY = "image/transform"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "image": "IMAGE",
        "bboxes": "BOUNDING_BOX",
        "output_width": "INT",
        "output_height": "INT",
        "padding": "INT",
        "keep_aspect": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        bboxes: tuple[str, int],
        output_width: int | tuple[str, int] = 512,
        output_height: int | tuple[str, int] = 512,
        padding: int | tuple[str, int] = 0,
        keep_aspect: CropByBBoxes_Keep_Aspects | tuple[str, int] = "stretch",
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.bboxes = bboxes
        self.output_width = output_width
        self.output_height = output_height
        self.padding = padding
        self.keep_aspect = keep_aspect
        self.__validate()

    def __validate(self) -> None:
        validate_range("output_width", self.output_width, 64, 4096)
        validate_range("output_height", self.output_height, 64, 4096)
        validate_range("padding", self.padding, 0, 1024)
        validate_choice("keep_aspect", self.keep_aspect, CROPBYBBOXES_KEEP_ASPECTS)


ResizeImageMaskNode_Resize_Types = Literal[
    "scale dimensions",
    "scale by multiplier",
    "scale longer dimension",
    "scale shorter dimension",
    "scale width",
    "scale height",
    "scale total pixels",
    "match size",
    "scale to multiple",
]


ResizeImageMaskNode_Scale_Methods = Literal[
    "nearest-exact", "bilinear", "area", "bicubic", "lanczos"
]


RESIZEIMAGEMASKNODE_SCALE_METHODS = frozenset(
    get_args(ResizeImageMaskNode_Scale_Methods)
)


ResizeImageMaskNode_Resize_Type_Crops = Literal["disabled", "center"]


class ResizeImageMaskNode(DynamicInputs, BaseNode):
    __slots__ = (
        "input",
        "resize_type",
        "resize_type_width",
        "resize_type_height",
        "resize_type_crop",
        "resize_type_multiplier",
        "resize_type_longer_size",
        "resize_type_shorter_size",
        "resize_type_megapixels",
        "resize_type_match",
        "resize_type_multiple",
        "scale_method",
    )
    CATEGORY = "image/transform"
    OUTPUTS = {"resized": 0}
    RETURN_TYPES = ("COMFY_MATCHTYPE_V3",)
    INPUT_NAMES = {
        "resize_type_width": "resize_type.width",
        "resize_type_height": "resize_type.height",
        "resize_type_crop": "resize_type.crop",
        "resize_type_multiplier": "resize_type.multiplier",
        "resize_type_longer_size": "resize_type.longer_size",
        "resize_type_shorter_size": "resize_type.shorter_size",
        "resize_type_megapixels": "resize_type.megapixels",
        "resize_type_match": "resize_type.match",
        "resize_type_multiple": "resize_type.multiple",
    }
    INPUT_SOCKET_TYPES = {
        "input": "COMFY_MATCHTYPE_V3",
        "resize_type": "COMFY_DYNAMICCOMBO_V3",
        "resize_type_width": "INT",
        "resize_type_height": "INT",
        "resize_type_crop": "COMBO",
        "resize_type_multiplier": "FLOAT",
        "resize_type_longer_size": "INT",
        "resize_type_shorter_size": "INT",
        "resize_type_megapixels": "FLOAT",
        "resize_type_match": "IMAGE,MASK",
        "resize_type_multiple": "INT",
        "scale_method": "COMBO",
    }
    DYNAMIC_INPUTS = {
        "resize_type": Combo(
            {
                "scale dimensions": {
                    "resize_type_width": Number(default=512, low=0, high=16384),
                    "resize_type_height": Number(default=512, low=0, high=16384),
                    "resize_type_crop": Choice(
                        ("disabled", "center"), default="center"
                    ),
                },
                "scale by multiplier": {
                    "resize_type_multiplier": Number(default=1.0, low=0.01, high=8.0)
                },
                "scale longer dimension": {
                    "resize_type_longer_size": Number(default=512, low=0, high=16384)
                },
                "scale shorter dimension": {
                    "resize_type_shorter_size": Number(default=512, low=0, high=16384)
                },
                "scale width": {
                    "resize_type_width": Number(default=512, low=0, high=16384)
                },
                "scale height": {
                    "resize_type_height": Number(default=512, low=0, high=16384)
                },
                "scale total pixels": {
                    "resize_type_megapixels": Number(default=1.0, low=0.01, high=16.0)
                },
                "match size": {
                    "resize_type_match": Link(),
                    "resize_type_crop": Choice(
                        ("disabled", "center"), default="center"
                    ),
                },
                "scale to multiple": {
                    "resize_type_multiple": Number(default=8, low=1, high=16384)
                },
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        input: tuple[str, int],
        resize_type: ResizeImageMaskNode_Resize_Types = "scale dimensions",
        resize_type_width: int | tuple[str, int] | None = None,
        resize_type_height: int | tuple[str, int] | None = None,
        resize_type_crop: (
            ResizeImageMaskNode_Resize_Type_Crops | tuple[str, int] | None
        ) = None,
        resize_type_multiplier: float | tuple[str, int] | None = None,
        resize_type_longer_size: int | tuple[str, int] | None = None,
        resize_type_shorter_size: int | tuple[str, int] | None = None,
        resize_type_megapixels: float | tuple[str, int] | None = None,
        resize_type_match: tuple[str, int] | None = None,
        resize_type_multiple: int | tuple[str, int] | None = None,
        scale_method: ResizeImageMaskNode_Scale_Methods | tuple[str, int] = "area",
    ) -> None:
        super().__init__(node_id=node_id)
        self.input = input
        self.resize_type = resize_type
        if resize_type_width is not None:
            self.resize_type_width = resize_type_width
        if resize_type_height is not None:
            self.resize_type_height = resize_type_height
        if resize_type_crop is not None:
            self.resize_type_crop = resize_type_crop
        if resize_type_multiplier is not None:
            self.resize_type_multiplier = resize_type_multiplier
        if resize_type_longer_size is not None:
            self.resize_type_longer_size = resize_type_longer_size
        if resize_type_shorter_size is not None:
            self.resize_type_shorter_size = resize_type_shorter_size
        if resize_type_megapixels is not None:
            self.resize_type_megapixels = resize_type_megapixels
        if resize_type_match is not None:
            self.resize_type_match = resize_type_match
        if resize_type_multiple is not None:
            self.resize_type_multiple = resize_type_multiple
        self.scale_method = scale_method
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "scale_method", self.scale_method, RESIZEIMAGEMASKNODE_SCALE_METHODS
        )
        validate_dynamic(self)
