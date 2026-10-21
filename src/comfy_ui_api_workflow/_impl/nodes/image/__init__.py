from typing import Any, Literal, get_args

from ..base import BaseNode
from ..validation import (
    validate_bool,
    validate_choice,
    validate_not_empty,
    validate_range,
)
from ..dynamic import Choice, Combo, DynamicInputs, Number, validate_dynamic


class SaveImage(BaseNode):
    __slots__ = ("images", "filename_prefix")
    CATEGORY = "image"
    OUTPUTS = {"images": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"images": "IMAGE", "filename_prefix": "STRING"}

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        filename_prefix: str | tuple[str, int] = "ComfyUI",
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.filename_prefix = filename_prefix


class PreviewImage(BaseNode):
    __slots__ = ("images",)
    CATEGORY = "image"
    OUTPUTS = {"images": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"images": "IMAGE"}

    def __init__(self, node_id: str, images: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.images = images


class LoadImage(BaseNode):
    __slots__ = ("image",)
    CATEGORY = "image"
    OUTPUTS = {"IMAGE": 0, "MASK": 1}
    RETURN_TYPES = ("IMAGE", "MASK")
    INPUT_SOCKET_TYPES = {"image": "V1 COMBO"}

    def __init__(self, node_id: str, image: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("image", self.image)


LoadImageMask_Channels = Literal["alpha", "red", "green", "blue"]
LOADIMAGEMASK_CHANNELS = frozenset(get_args(LoadImageMask_Channels))


class LoadImageMask(BaseNode):
    __slots__ = ("image", "channel")
    CATEGORY = "image"
    OUTPUTS = {"MASK": 0}
    RETURN_TYPES = ("MASK",)
    INPUT_SOCKET_TYPES = {"image": "V1 COMBO", "channel": "V1 COMBO"}

    def __init__(
        self,
        node_id: str,
        image: str | tuple[str, int],
        channel: LoadImageMask_Channels | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.channel = channel
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("image", self.image)
        validate_choice("channel", self.channel, LOADIMAGEMASK_CHANNELS)


class LoadImageOutput(BaseNode):
    __slots__ = ("image",)
    CATEGORY = "image"
    OUTPUTS = {"IMAGE": 0, "MASK": 1}
    RETURN_TYPES = ("IMAGE", "MASK")
    INPUT_SOCKET_TYPES = {"image": "COMBO"}

    def __init__(self, node_id: str, image: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("image", self.image)


class EmptyImage(BaseNode):
    __slots__ = ("width", "height", "batch_size", "color")
    CATEGORY = "image"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "width": "INT",
        "height": "INT",
        "batch_size": "INT",
        "color": "INT",
    }

    def __init__(
        self,
        node_id: str,
        width: int | tuple[str, int] = 512,
        height: int | tuple[str, int] = 512,
        batch_size: int | tuple[str, int] = 1,
        color: int | tuple[str, int] = 0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.color = color
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 1, 16_384)
        validate_range("height", self.height, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)
        validate_range("color", self.color, 0, 16_777_215)


SaveAnimatedWEBP_Methods = Literal["default", "fastest", "slowest"]
SAVEANIMATEDWEBP_METHODS = frozenset(get_args(SaveAnimatedWEBP_Methods))


class SaveAnimatedWEBP(BaseNode):
    __slots__ = ("images", "method", "filename_prefix", "fps", "lossless", "quality")
    CATEGORY = "image"
    OUTPUTS = {"images": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "images": "IMAGE",
        "filename_prefix": "STRING",
        "fps": "FLOAT",
        "lossless": "BOOLEAN",
        "quality": "INT",
        "method": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        method: SaveAnimatedWEBP_Methods | tuple[str, int],
        filename_prefix: str | tuple[str, int] = "ComfyUI",
        fps: float | tuple[str, int] = 6.0,
        lossless: bool | tuple[str, int] = True,
        quality: int | tuple[str, int] = 80,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.method = method
        self.filename_prefix = filename_prefix
        self.fps = fps
        self.lossless = lossless
        self.quality = quality
        self.__validate()

    def __validate(self) -> None:
        validate_choice("method", self.method, SAVEANIMATEDWEBP_METHODS)
        validate_range("fps", self.fps, 0.01, 1000.0)
        validate_bool("lossless", self.lossless)
        validate_range("quality", self.quality, 0, 100)


class SaveAnimatedPNG(BaseNode):
    __slots__ = ("images", "filename_prefix", "fps", "compress_level")
    CATEGORY = "image"
    OUTPUTS = {"images": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "images": "IMAGE",
        "filename_prefix": "STRING",
        "fps": "FLOAT",
        "compress_level": "INT",
    }

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        filename_prefix: str | tuple[str, int] = "ComfyUI",
        fps: float | tuple[str, int] = 6.0,
        compress_level: int | tuple[str, int] = 4,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.filename_prefix = filename_prefix
        self.fps = fps
        self.compress_level = compress_level
        self.__validate()

    def __validate(self) -> None:
        validate_range("fps", self.fps, 0.01, 1000.0)
        validate_range("compress_level", self.compress_level, 0, 9)


class SaveSVGNode(BaseNode):
    __slots__ = ("svg", "filename_prefix")
    CATEGORY = "image"
    OUTPUTS = {"svg": 0}
    RETURN_TYPES = ("SVG",)
    INPUT_SOCKET_TYPES = {"svg": "SVG", "filename_prefix": "STRING"}

    def __init__(
        self,
        node_id: str,
        svg: tuple[str, int],
        filename_prefix: str | tuple[str, int] = "svg/ComfyUI",
    ) -> None:
        super().__init__(node_id=node_id)
        self.svg = svg
        self.filename_prefix = filename_prefix


class GetImageSize(BaseNode):
    __slots__ = ("image",)
    CATEGORY = "image"
    OUTPUTS = {"width": 0, "height": 1, "batch_size": 2}
    RETURN_TYPES = ("INT", "INT", "INT")
    INPUT_SOCKET_TYPES = {"image": "IMAGE"}

    def __init__(self, node_id: str, image: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.image = image


class LoadImageDataSetFromFolder(BaseNode):
    __slots__ = ("folder",)
    CATEGORY = "image"
    OUTPUTS = {"images": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"folder": "COMBO"}

    def __init__(self, node_id: str, folder: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.folder = folder
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("folder", self.folder)


class LoadImageTextDataSetFromFolder(BaseNode):
    __slots__ = ("folder",)
    CATEGORY = "image"
    OUTPUTS = {"images": 0, "texts": 1}
    RETURN_TYPES = ("IMAGE", "STRING")
    INPUT_SOCKET_TYPES = {"folder": "COMBO"}

    def __init__(self, node_id: str, folder: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.folder = folder
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("folder", self.folder)


SaveImageTextDataSetToFolder_Modes = Literal["overwrite", "increment"]
SAVEIMAGETEXTDATASETTOFOLDER_MODES = frozenset(
    get_args(SaveImageTextDataSetToFolder_Modes)
)


class SaveImageTextDataSetToFolder(BaseNode):
    __slots__ = ("images", "folder_name", "filename_prefix", "mode", "texts")
    CATEGORY = "image"
    INPUT_SOCKET_TYPES = {
        "images": "IMAGE",
        "folder_name": "STRING",
        "filename_prefix": "STRING",
        "mode": "COMBO",
        "texts": "STRING",
    }

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        folder_name: str | tuple[str, int] = "dataset",
        filename_prefix: str | tuple[str, int] = "image",
        mode: SaveImageTextDataSetToFolder_Modes | tuple[str, int] = "overwrite",
        texts: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.folder_name = folder_name
        self.filename_prefix = filename_prefix
        self.mode = mode
        if texts is not None:
            self.texts = texts
        self.__validate()

    def __validate(self) -> None:
        validate_choice("mode", self.mode, SAVEIMAGETEXTDATASETTOFOLDER_MODES)


class WebcamCapture(BaseNode):
    __slots__ = ("image", "width", "height", "capture_on_queue")
    CATEGORY = "image"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "image": "WEBCAM",
        "width": "INT",
        "height": "INT",
        "capture_on_queue": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        image: Any,
        width: int | tuple[str, int] = 0,
        height: int | tuple[str, int] = 0,
        capture_on_queue: bool | tuple[str, int] = True,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.width = width
        self.height = height
        self.capture_on_queue = capture_on_queue
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 0, 16_384)
        validate_range("height", self.height, 0, 16_384)
        validate_bool("capture_on_queue", self.capture_on_queue)


class ImageCompare(BaseNode):
    __slots__ = ("compare_view", "image_a", "image_b")
    CATEGORY = "image"
    INPUT_SOCKET_TYPES = {
        "compare_view": "IMAGECOMPARE",
        "image_a": "IMAGE",
        "image_b": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        compare_view: Any,
        image_a: tuple[str, int] | None = None,
        image_b: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.compare_view = compare_view
        if image_a is not None:
            self.image_a = image_a
        if image_b is not None:
            self.image_b = image_b


class Painter(BaseNode):
    __slots__ = ("mask", "width", "height", "bg_color", "image")
    CATEGORY = "image"
    OUTPUTS = {"IMAGE": 0, "MASK": 1}
    RETURN_TYPES = ("IMAGE", "MASK")
    INPUT_SOCKET_TYPES = {
        "mask": "STRING",
        "width": "INT",
        "height": "INT",
        "bg_color": "COLOR",
        "image": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        mask: Any = "",
        width: int | tuple[str, int] = 512,
        height: int | tuple[str, int] = 512,
        bg_color: Any = "#000000",
        image: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mask = mask
        self.width = width
        self.height = height
        self.bg_color = bg_color
        if image is not None:
            self.image = image
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 64, 4096)
        validate_range("height", self.height, 64, 4096)


class ImageCompositor(BaseNode):
    __slots__ = ("layers", "compositor")
    CATEGORY = "image"
    OUTPUTS = {"IMAGE": 0, "MASK": 1}
    RETURN_TYPES = ("IMAGE", "MASK")
    INPUT_SOCKET_TYPES = {"layers": "LAYERS", "compositor": "COMPOSITOR"}

    def __init__(
        self, node_id: str, layers: tuple[str, int], compositor: Any = None
    ) -> None:
        super().__init__(node_id=node_id)
        self.layers = layers
        self.compositor = {} if compositor is None else compositor  # type: ignore


AddLayer_Blend_Modes = Literal[
    "normal",
    "multiply",
    "screen",
    "overlay",
    "darken",
    "lighten",
    "color-dodge",
    "color-burn",
    "hard-light",
    "soft-light",
    "difference",
    "exclusion",
    "linear-dodge",
    "linear-burn",
    "vivid-light",
    "pin-light",
    "linear-light",
    "hard-mix",
    "subtract",
    "divide",
    "grain-extract",
    "grain-merge",
    "hue",
    "saturation",
    "color",
    "luminosity",
]
ADDLAYER_BLEND_MODES = frozenset(get_args(AddLayer_Blend_Modes))


class AddLayer(BaseNode):
    __slots__ = (
        "image",
        "layers",
        "mask",
        "name",
        "x",
        "y",
        "opacity",
        "blend_mode",
        "rotation",
        "width",
        "height",
        "z_index",
        "flip_h",
        "flip_v",
    )
    CATEGORY = "image"
    OUTPUTS = {"LAYERS": 0}
    RETURN_TYPES = ("LAYERS",)
    INPUT_SOCKET_TYPES = {
        "image": "IMAGE",
        "layers": "LAYERS",
        "mask": "MASK",
        "name": "STRING",
        "x": "INT",
        "y": "INT",
        "opacity": "FLOAT",
        "blend_mode": "COMBO",
        "rotation": "FLOAT",
        "width": "INT",
        "height": "INT",
        "z_index": "INT",
        "flip_h": "BOOLEAN",
        "flip_v": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        layers: tuple[str, int] | None = None,
        mask: tuple[str, int] | None = None,
        name: str | tuple[str, int] = "",
        x: int | tuple[str, int] = 0,
        y: int | tuple[str, int] = 0,
        opacity: float | tuple[str, int] = 1.0,
        blend_mode: AddLayer_Blend_Modes | tuple[str, int] = "normal",
        rotation: float | tuple[str, int] = 0.0,
        width: int | tuple[str, int] = 0,
        height: int | tuple[str, int] = 0,
        z_index: int | tuple[str, int] = 0,
        flip_h: bool | tuple[str, int] = False,
        flip_v: bool | tuple[str, int] = False,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        if layers is not None:
            self.layers = layers
        if mask is not None:
            self.mask = mask
        self.name = name
        self.x = x
        self.y = y
        self.opacity = opacity
        self.blend_mode = blend_mode
        self.rotation = rotation
        self.width = width
        self.height = height
        self.z_index = z_index
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.__validate()

    def __validate(self) -> None:
        validate_range("x", self.x, -16_384, 16_384)
        validate_range("y", self.y, -16_384, 16_384)
        validate_range("opacity", self.opacity, 0.0, 1.0)
        validate_choice("blend_mode", self.blend_mode, ADDLAYER_BLEND_MODES)
        validate_range("rotation", self.rotation, -360.0, 360.0)
        validate_range("width", self.width, 0, 16_384)
        validate_range("height", self.height, 0, 16_384)
        validate_range("z_index", self.z_index, -1000, 1000)
        validate_bool("flip_h", self.flip_h)
        validate_bool("flip_v", self.flip_v)


class LayersFromBoundingBoxes(BaseNode):
    __slots__ = (
        "image",
        "bboxes",
        "mask",
        "layers",
        "crop_to_content",
        "canvas_width",
        "canvas_height",
    )
    CATEGORY = "image"
    OUTPUTS = {"LAYERS": 0}
    RETURN_TYPES = ("LAYERS",)
    INPUT_SOCKET_TYPES = {
        "image": "IMAGE",
        "bboxes": "BOUNDING_BOX,ARRAY,STRING",
        "mask": "MASK",
        "layers": "LAYERS",
        "crop_to_content": "BOOLEAN",
        "canvas_width": "INT",
        "canvas_height": "INT",
    }

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        bboxes: tuple[str, int],
        mask: tuple[str, int] | None = None,
        layers: tuple[str, int] | None = None,
        crop_to_content: bool | tuple[str, int] = True,
        canvas_width: int | tuple[str, int] = 0,
        canvas_height: int | tuple[str, int] = 0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.bboxes = bboxes
        if mask is not None:
            self.mask = mask
        if layers is not None:
            self.layers = layers
        self.crop_to_content = crop_to_content
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.__validate()

    def __validate(self) -> None:
        validate_bool("crop_to_content", self.crop_to_content)
        validate_range("canvas_width", self.canvas_width, 0, 16_384)
        validate_range("canvas_height", self.canvas_height, 0, 16_384)


class SaveImageWebsocket(BaseNode):
    __slots__ = ("images",)
    CATEGORY = "image"
    INPUT_SOCKET_TYPES = {"images": "IMAGE"}

    def __init__(self, node_id: str, images: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.images = images


SaveImageAdvanced_Formats = Literal["png", "exr", "avif"]


SaveImageAdvanced_Format_Bit_Depths = Literal[
    "8-bit",
    "16-bit",
    "32-bit float",
    "16-bit float",
    "auto",
    "8-bit YUV420",
    "10-bit YUV420",
]


SaveImageAdvanced_Format_Input_Color_Spaces = Literal["sRGB", "HDR", "linear", "HDR PQ"]


SaveImageAdvanced_Format_Save_Modes = Literal["still images", "animated"]


class SaveImageAdvanced(DynamicInputs, BaseNode):
    __slots__ = (
        "images",
        "filename_prefix",
        "format",
        "format_bit_depth",
        "format_input_color_space",
        "format_crf",
        "format_save_mode",
        "format_save_mode_fps",
        "format_save_mode_loop_count",
    )
    CATEGORY = "image"
    OUTPUTS = {"images": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_NAMES = {
        "format_bit_depth": "format.bit_depth",
        "format_input_color_space": "format.input_color_space",
        "format_crf": "format.crf",
        "format_save_mode": "format.save_mode",
        "format_save_mode_fps": "format.save_mode.fps",
        "format_save_mode_loop_count": "format.save_mode.loop_count",
    }
    INPUT_SOCKET_TYPES = {
        "images": "IMAGE",
        "filename_prefix": "STRING",
        "format": "COMFY_DYNAMICCOMBO_V3",
        "format_bit_depth": "COMBO",
        "format_input_color_space": "COMBO",
        "format_crf": "INT",
        "format_save_mode": "COMFY_DYNAMICCOMBO_V3",
        "format_save_mode_fps": "FLOAT",
        "format_save_mode_loop_count": "INT",
    }
    DYNAMIC_INPUTS = {
        "format": Combo(
            {
                "png": {
                    "format_bit_depth": Choice(("8-bit", "16-bit"), default="8-bit"),
                    "format_input_color_space": Choice(("sRGB",), default="sRGB"),
                },
                "exr": {
                    "format_bit_depth": Choice(
                        ("32-bit float", "16-bit float"), default="16-bit float"
                    ),
                    "format_input_color_space": Choice(
                        ("sRGB", "HDR", "linear"), default="sRGB"
                    ),
                },
                "avif": {
                    "format_bit_depth": Choice(
                        ("auto", "8-bit YUV420", "10-bit YUV420"), default="auto"
                    ),
                    "format_input_color_space": Choice(
                        ("sRGB", "HDR", "HDR PQ"), default="sRGB"
                    ),
                    "format_crf": Number(default=18, low=1, high=63),
                    "format_save_mode": Combo(
                        {
                            "still images": {},
                            "animated": {
                                "format_save_mode_fps": Number(
                                    default=6.0, low=0.01, high=1000.0
                                ),
                                "format_save_mode_loop_count": Number(
                                    default=0, low=0, high=1000
                                ),
                            },
                        }
                    ),
                },
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        filename_prefix: str | tuple[str, int] = "ComfyUI",
        format: SaveImageAdvanced_Formats = "png",
        format_bit_depth: (
            SaveImageAdvanced_Format_Bit_Depths | tuple[str, int] | None
        ) = None,
        format_input_color_space: (
            SaveImageAdvanced_Format_Input_Color_Spaces | tuple[str, int] | None
        ) = None,
        format_crf: int | tuple[str, int] | None = None,
        format_save_mode: SaveImageAdvanced_Format_Save_Modes | None = None,
        format_save_mode_fps: float | tuple[str, int] | None = None,
        format_save_mode_loop_count: int | tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.filename_prefix = filename_prefix
        self.format = format
        if format_bit_depth is not None:
            self.format_bit_depth = format_bit_depth
        if format_input_color_space is not None:
            self.format_input_color_space = format_input_color_space
        if format_crf is not None:
            self.format_crf = format_crf
        if format_save_mode is not None:
            self.format_save_mode = format_save_mode
        if format_save_mode_fps is not None:
            self.format_save_mode_fps = format_save_mode_fps
        if format_save_mode_loop_count is not None:
            self.format_save_mode_loop_count = format_save_mode_loop_count
        self.__validate()

    def __validate(self) -> None:
        validate_dynamic(self)


from . import (
    adjustments,
    background_removal,
    batch,
    color,
    compositing,
    detection,
    filters,
    geometry_estimation,
    mask,
    post_processors,
    pre_processors,
    shader,
    transform,
    upscaling,
    video,
)
