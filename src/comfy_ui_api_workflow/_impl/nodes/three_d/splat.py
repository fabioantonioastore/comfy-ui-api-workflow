from typing import Any, Literal, get_args

from ..base import BaseNode
from ..validation import validate_choice, validate_range

SplatToFile3D_Formats = Literal["ply", "ksplat", "spz"]
SPLATTOFILE3D_FORMATS = frozenset(get_args(SplatToFile3D_Formats))


class SplatToFile3D(BaseNode):
    __slots__ = ("splat", "format")
    CATEGORY = "3d/splat"
    OUTPUTS = {"model_3d": 0}
    RETURN_TYPES = ("FILE_3D_SPLAT_ANY",)
    INPUT_SOCKET_TYPES = {"splat": "SPLAT", "format": "COMBO"}

    def __init__(
        self,
        node_id: str,
        splat: tuple[str, int],
        format: SplatToFile3D_Formats | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.splat = splat
        self.format = format
        self.__validate()

    def __validate(self) -> None:
        validate_choice("format", self.format, SPLATTOFILE3D_FORMATS)


class File3DToSplat(BaseNode):
    __slots__ = ("model_3d",)
    CATEGORY = "3d/splat"
    OUTPUTS = {"splat": 0}
    RETURN_TYPES = ("SPLAT",)
    INPUT_SOCKET_TYPES = {
        "model_3d": "FILE_3D,FILE_3D_SPLAT_ANY,FILE_3D_PLY,FILE_3D_SPLAT,FILE_3D_KSPLAT,FILE_3D_SPZ"
    }

    def __init__(self, node_id: str, model_3d: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.model_3d = model_3d


RenderSplat_Render_Styles = Literal["color", "clay", "depth", "normal"]
RENDERSPLAT_RENDER_STYLES = frozenset(get_args(RenderSplat_Render_Styles))


class RenderSplat(BaseNode):
    __slots__ = (
        "splat",
        "render_style",
        "width",
        "height",
        "frames",
        "splat_scale",
        "sharpen",
        "headlight_shading",
        "opacity_threshold",
        "background",
        "bg_image",
        "camera_info",
    )
    CATEGORY = "3d/splat"
    OUTPUTS = {"image": 0, "mask": 1}
    RETURN_TYPES = ("IMAGE", "MASK")
    INPUT_SOCKET_TYPES = {
        "splat": "SPLAT",
        "width": "INT",
        "height": "INT",
        "frames": "INT",
        "splat_scale": "FLOAT",
        "sharpen": "FLOAT",
        "headlight_shading": "FLOAT",
        "opacity_threshold": "FLOAT",
        "render_style": "COMBO",
        "background": "COLOR",
        "bg_image": "IMAGE",
        "camera_info": "LOAD3D_CAMERA",
    }

    def __init__(
        self,
        node_id: str,
        splat: tuple[str, int],
        render_style: RenderSplat_Render_Styles | tuple[str, int],
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
        frames: int | tuple[str, int] = 1,
        splat_scale: float | tuple[str, int] = 1.0,
        sharpen: float | tuple[str, int] = 2.0,
        headlight_shading: float | tuple[str, int] = 0.0,
        opacity_threshold: float | tuple[str, int] = 0.0,
        background: Any = "#000000",
        bg_image: tuple[str, int] | None = None,
        camera_info: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.splat = splat
        self.render_style = render_style
        self.width = width
        self.height = height
        self.frames = frames
        self.splat_scale = splat_scale
        self.sharpen = sharpen
        self.headlight_shading = headlight_shading
        self.opacity_threshold = opacity_threshold
        self.background = background
        if bg_image is not None:
            self.bg_image = bg_image
        if camera_info is not None:
            self.camera_info = camera_info
        self.__validate()

    def __validate(self) -> None:
        validate_choice("render_style", self.render_style, RENDERSPLAT_RENDER_STYLES)
        validate_range("width", self.width, 64, 2048)
        validate_range("height", self.height, 64, 2048)
        validate_range("frames", self.frames, -240, 240)
        validate_range("splat_scale", self.splat_scale, 0.1, 5.0)
        validate_range("sharpen", self.sharpen, 1.0, 8.0)
        validate_range("headlight_shading", self.headlight_shading, 0.0, 3.0)
        validate_range("opacity_threshold", self.opacity_threshold, 0.0, 1.0)


class TransformSplat(BaseNode):
    __slots__ = (
        "splat",
        "translate_x",
        "translate_y",
        "translate_z",
        "rotate_x",
        "rotate_y",
        "rotate_z",
        "scale_x",
        "scale_y",
        "scale_z",
    )
    CATEGORY = "3d/splat"
    OUTPUTS = {"splat": 0}
    RETURN_TYPES = ("SPLAT",)
    INPUT_SOCKET_TYPES = {
        "splat": "SPLAT",
        "translate_x": "FLOAT",
        "translate_y": "FLOAT",
        "translate_z": "FLOAT",
        "rotate_x": "FLOAT",
        "rotate_y": "FLOAT",
        "rotate_z": "FLOAT",
        "scale_x": "FLOAT",
        "scale_y": "FLOAT",
        "scale_z": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        splat: tuple[str, int],
        translate_x: float | tuple[str, int] = 0.0,
        translate_y: float | tuple[str, int] = 0.0,
        translate_z: float | tuple[str, int] = 0.0,
        rotate_x: float | tuple[str, int] = 0.0,
        rotate_y: float | tuple[str, int] = 0.0,
        rotate_z: float | tuple[str, int] = 0.0,
        scale_x: float | tuple[str, int] = 1.0,
        scale_y: float | tuple[str, int] = 1.0,
        scale_z: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.splat = splat
        self.translate_x = translate_x
        self.translate_y = translate_y
        self.translate_z = translate_z
        self.rotate_x = rotate_x
        self.rotate_y = rotate_y
        self.rotate_z = rotate_z
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z
        self.__validate()

    def __validate(self) -> None:
        validate_range("translate_x", self.translate_x, -100.0, 100.0)
        validate_range("translate_y", self.translate_y, -100.0, 100.0)
        validate_range("translate_z", self.translate_z, -100.0, 100.0)
        validate_range("rotate_x", self.rotate_x, -360.0, 360.0)
        validate_range("rotate_y", self.rotate_y, -360.0, 360.0)
        validate_range("rotate_z", self.rotate_z, -360.0, 360.0)
        validate_range("scale_x", self.scale_x, 0.01, 100.0)
        validate_range("scale_y", self.scale_y, 0.01, 100.0)
        validate_range("scale_z", self.scale_z, 0.01, 100.0)


class GetSplatCount(BaseNode):
    __slots__ = ("splat",)
    CATEGORY = "3d/splat"
    OUTPUTS = {"splat": 0, "count": 1}
    RETURN_TYPES = ("SPLAT", "INT")
    INPUT_SOCKET_TYPES = {"splat": "SPLAT"}

    def __init__(self, node_id: str, splat: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.splat = splat


class SplatToMesh(BaseNode):
    __slots__ = (
        "splat",
        "resolution",
        "kernel",
        "smooth",
        "level",
        "min_component",
        "min_opacity",
        "color_sharpen",
    )
    CATEGORY = "3d/splat"
    OUTPUTS = {"mesh": 0}
    RETURN_TYPES = ("MESH",)
    INPUT_SOCKET_TYPES = {
        "splat": "SPLAT",
        "resolution": "INT",
        "kernel": "INT",
        "smooth": "INT",
        "level": "FLOAT",
        "min_component": "INT",
        "min_opacity": "FLOAT",
        "color_sharpen": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        splat: tuple[str, int],
        resolution: int | tuple[str, int] = 384,
        kernel: int | tuple[str, int] = 5,
        smooth: int | tuple[str, int] = 0,
        level: float | tuple[str, int] = 0.4,
        min_component: int | tuple[str, int] = 500,
        min_opacity: float | tuple[str, int] = 0.02,
        color_sharpen: float | tuple[str, int] = 2.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.splat = splat
        self.resolution = resolution
        self.kernel = kernel
        self.smooth = smooth
        self.level = level
        self.min_component = min_component
        self.min_opacity = min_opacity
        self.color_sharpen = color_sharpen
        self.__validate()

    def __validate(self) -> None:
        validate_range("resolution", self.resolution, 64, 768)
        validate_range("kernel", self.kernel, 1, 8)
        validate_range("smooth", self.smooth, 0, 60)
        validate_range("level", self.level, 0.0, 2.0)
        validate_range("min_component", self.min_component, 0, 100_000)
        validate_range("min_opacity", self.min_opacity, 0.0, 1.0)
        validate_range("color_sharpen", self.color_sharpen, 1.0, 8.0)


class MergeSplat(BaseNode):
    __slots__ = (
        "splats_splat0",
        "splats_splat1",
        "splats_splat2",
        "splats_splat3",
        "splats_splat4",
        "splats_splat5",
        "splats_splat6",
        "splats_splat7",
        "splats_splat8",
        "splats_splat9",
        "splats_splat10",
        "splats_splat11",
        "splats_splat12",
        "splats_splat13",
        "splats_splat14",
        "splats_splat15",
        "splats_splat16",
        "splats_splat17",
        "splats_splat18",
        "splats_splat19",
        "splats_splat20",
        "splats_splat21",
        "splats_splat22",
        "splats_splat23",
        "splats_splat24",
        "splats_splat25",
        "splats_splat26",
        "splats_splat27",
        "splats_splat28",
        "splats_splat29",
        "splats_splat30",
        "splats_splat31",
    )
    CATEGORY = "3d/splat"
    OUTPUTS = {"splat": 0}
    RETURN_TYPES = ("SPLAT",)
    INPUT_NAMES = {
        "splats_splat0": "splats.splat0",
        "splats_splat1": "splats.splat1",
        "splats_splat2": "splats.splat2",
        "splats_splat3": "splats.splat3",
        "splats_splat4": "splats.splat4",
        "splats_splat5": "splats.splat5",
        "splats_splat6": "splats.splat6",
        "splats_splat7": "splats.splat7",
        "splats_splat8": "splats.splat8",
        "splats_splat9": "splats.splat9",
        "splats_splat10": "splats.splat10",
        "splats_splat11": "splats.splat11",
        "splats_splat12": "splats.splat12",
        "splats_splat13": "splats.splat13",
        "splats_splat14": "splats.splat14",
        "splats_splat15": "splats.splat15",
        "splats_splat16": "splats.splat16",
        "splats_splat17": "splats.splat17",
        "splats_splat18": "splats.splat18",
        "splats_splat19": "splats.splat19",
        "splats_splat20": "splats.splat20",
        "splats_splat21": "splats.splat21",
        "splats_splat22": "splats.splat22",
        "splats_splat23": "splats.splat23",
        "splats_splat24": "splats.splat24",
        "splats_splat25": "splats.splat25",
        "splats_splat26": "splats.splat26",
        "splats_splat27": "splats.splat27",
        "splats_splat28": "splats.splat28",
        "splats_splat29": "splats.splat29",
        "splats_splat30": "splats.splat30",
        "splats_splat31": "splats.splat31",
    }
    INPUT_SOCKET_TYPES = {
        "splats_splat0": "SPLAT",
        "splats_splat1": "SPLAT",
        "splats_splat2": "SPLAT",
        "splats_splat3": "SPLAT",
        "splats_splat4": "SPLAT",
        "splats_splat5": "SPLAT",
        "splats_splat6": "SPLAT",
        "splats_splat7": "SPLAT",
        "splats_splat8": "SPLAT",
        "splats_splat9": "SPLAT",
        "splats_splat10": "SPLAT",
        "splats_splat11": "SPLAT",
        "splats_splat12": "SPLAT",
        "splats_splat13": "SPLAT",
        "splats_splat14": "SPLAT",
        "splats_splat15": "SPLAT",
        "splats_splat16": "SPLAT",
        "splats_splat17": "SPLAT",
        "splats_splat18": "SPLAT",
        "splats_splat19": "SPLAT",
        "splats_splat20": "SPLAT",
        "splats_splat21": "SPLAT",
        "splats_splat22": "SPLAT",
        "splats_splat23": "SPLAT",
        "splats_splat24": "SPLAT",
        "splats_splat25": "SPLAT",
        "splats_splat26": "SPLAT",
        "splats_splat27": "SPLAT",
        "splats_splat28": "SPLAT",
        "splats_splat29": "SPLAT",
        "splats_splat30": "SPLAT",
        "splats_splat31": "SPLAT",
    }

    def __init__(
        self,
        node_id: str,
        splats_splat0: tuple[str, int],
        splats_splat1: tuple[str, int],
        splats_splat2: tuple[str, int] | None = None,
        splats_splat3: tuple[str, int] | None = None,
        splats_splat4: tuple[str, int] | None = None,
        splats_splat5: tuple[str, int] | None = None,
        splats_splat6: tuple[str, int] | None = None,
        splats_splat7: tuple[str, int] | None = None,
        splats_splat8: tuple[str, int] | None = None,
        splats_splat9: tuple[str, int] | None = None,
        splats_splat10: tuple[str, int] | None = None,
        splats_splat11: tuple[str, int] | None = None,
        splats_splat12: tuple[str, int] | None = None,
        splats_splat13: tuple[str, int] | None = None,
        splats_splat14: tuple[str, int] | None = None,
        splats_splat15: tuple[str, int] | None = None,
        splats_splat16: tuple[str, int] | None = None,
        splats_splat17: tuple[str, int] | None = None,
        splats_splat18: tuple[str, int] | None = None,
        splats_splat19: tuple[str, int] | None = None,
        splats_splat20: tuple[str, int] | None = None,
        splats_splat21: tuple[str, int] | None = None,
        splats_splat22: tuple[str, int] | None = None,
        splats_splat23: tuple[str, int] | None = None,
        splats_splat24: tuple[str, int] | None = None,
        splats_splat25: tuple[str, int] | None = None,
        splats_splat26: tuple[str, int] | None = None,
        splats_splat27: tuple[str, int] | None = None,
        splats_splat28: tuple[str, int] | None = None,
        splats_splat29: tuple[str, int] | None = None,
        splats_splat30: tuple[str, int] | None = None,
        splats_splat31: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.splats_splat0 = splats_splat0
        self.splats_splat1 = splats_splat1
        if splats_splat2 is not None:
            self.splats_splat2 = splats_splat2
        if splats_splat3 is not None:
            self.splats_splat3 = splats_splat3
        if splats_splat4 is not None:
            self.splats_splat4 = splats_splat4
        if splats_splat5 is not None:
            self.splats_splat5 = splats_splat5
        if splats_splat6 is not None:
            self.splats_splat6 = splats_splat6
        if splats_splat7 is not None:
            self.splats_splat7 = splats_splat7
        if splats_splat8 is not None:
            self.splats_splat8 = splats_splat8
        if splats_splat9 is not None:
            self.splats_splat9 = splats_splat9
        if splats_splat10 is not None:
            self.splats_splat10 = splats_splat10
        if splats_splat11 is not None:
            self.splats_splat11 = splats_splat11
        if splats_splat12 is not None:
            self.splats_splat12 = splats_splat12
        if splats_splat13 is not None:
            self.splats_splat13 = splats_splat13
        if splats_splat14 is not None:
            self.splats_splat14 = splats_splat14
        if splats_splat15 is not None:
            self.splats_splat15 = splats_splat15
        if splats_splat16 is not None:
            self.splats_splat16 = splats_splat16
        if splats_splat17 is not None:
            self.splats_splat17 = splats_splat17
        if splats_splat18 is not None:
            self.splats_splat18 = splats_splat18
        if splats_splat19 is not None:
            self.splats_splat19 = splats_splat19
        if splats_splat20 is not None:
            self.splats_splat20 = splats_splat20
        if splats_splat21 is not None:
            self.splats_splat21 = splats_splat21
        if splats_splat22 is not None:
            self.splats_splat22 = splats_splat22
        if splats_splat23 is not None:
            self.splats_splat23 = splats_splat23
        if splats_splat24 is not None:
            self.splats_splat24 = splats_splat24
        if splats_splat25 is not None:
            self.splats_splat25 = splats_splat25
        if splats_splat26 is not None:
            self.splats_splat26 = splats_splat26
        if splats_splat27 is not None:
            self.splats_splat27 = splats_splat27
        if splats_splat28 is not None:
            self.splats_splat28 = splats_splat28
        if splats_splat29 is not None:
            self.splats_splat29 = splats_splat29
        if splats_splat30 is not None:
            self.splats_splat30 = splats_splat30
        if splats_splat31 is not None:
            self.splats_splat31 = splats_splat31
