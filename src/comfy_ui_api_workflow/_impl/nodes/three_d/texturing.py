from typing import Literal, get_args

from ..base import BaseNode
from ..validation import validate_bool, validate_choice, validate_range

UnwrapMesh_Segmenters = Literal["pec", "adaptive"]
UNWRAPMESH_SEGMENTERS = frozenset(get_args(UnwrapMesh_Segmenters))


class UnwrapMesh(BaseNode):
    __slots__ = ("mesh", "segmenter", "resolution", "padding", "weld_distance")
    CATEGORY = "3d/texturing"
    OUTPUTS = {"mesh": 0}
    RETURN_TYPES = ("MESH",)
    INPUT_SOCKET_TYPES = {
        "mesh": "MESH",
        "segmenter": "COMBO",
        "resolution": "INT",
        "padding": "INT",
        "weld_distance": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        mesh: tuple[str, int],
        segmenter: UnwrapMesh_Segmenters | tuple[str, int] = "pec",
        resolution: int | tuple[str, int] = 1024,
        padding: int | tuple[str, int] = 1,
        weld_distance: float | tuple[str, int] = 0.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mesh = mesh
        self.segmenter = segmenter
        self.resolution = resolution
        self.padding = padding
        self.weld_distance = weld_distance
        self.__validate()

    def __validate(self) -> None:
        validate_choice("segmenter", self.segmenter, UNWRAPMESH_SEGMENTERS)
        validate_range("resolution", self.resolution, 0, 8192)
        validate_range("padding", self.padding, 0, 16)
        validate_range("weld_distance", self.weld_distance, 0.0, 1.0)


class RenderUVAtlas(BaseNode):
    __slots__ = ("mesh", "resolution")
    CATEGORY = "3d/texturing"
    OUTPUTS = {"image": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"mesh": "MESH", "resolution": "INT"}

    def __init__(
        self,
        node_id: str,
        mesh: tuple[str, int],
        resolution: int | tuple[str, int] = 1024,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mesh = mesh
        self.resolution = resolution
        self.__validate()

    def __validate(self) -> None:
        validate_range("resolution", self.resolution, 64, 4096)


class BakeTextureFromVoxel(BaseNode):
    __slots__ = ("mesh", "voxel_colors", "texture_size", "reference_mesh")
    CATEGORY = "3d/texturing"
    OUTPUTS = {"base_color": 0, "metallic": 1, "roughness": 2}
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    INPUT_SOCKET_TYPES = {
        "mesh": "MESH",
        "voxel_colors": "VOXEL",
        "texture_size": "INT",
        "reference_mesh": "MESH",
    }

    def __init__(
        self,
        node_id: str,
        mesh: tuple[str, int],
        voxel_colors: tuple[str, int],
        texture_size: int | tuple[str, int] = 2048,
        reference_mesh: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mesh = mesh
        self.voxel_colors = voxel_colors
        self.texture_size = texture_size
        if reference_mesh is not None:
            self.reference_mesh = reference_mesh
        self.__validate()

    def __validate(self) -> None:
        validate_range("texture_size", self.texture_size, 64, 8192)


class MeshTextureToImage(BaseNode):
    __slots__ = ("mesh",)
    CATEGORY = "3d/texturing"
    OUTPUTS = {
        "base_color": 0,
        "metallic": 1,
        "roughness": 2,
        "occlusion": 3,
        "normal_map": 4,
    }
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    INPUT_SOCKET_TYPES = {"mesh": "MESH"}

    def __init__(self, node_id: str, mesh: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.mesh = mesh


class ApplyTextureToMesh(BaseNode):
    __slots__ = (
        "mesh",
        "base_color",
        "metallic",
        "roughness",
        "occlusion",
        "normal_map",
    )
    CATEGORY = "3d/texturing"
    OUTPUTS = {"mesh": 0}
    RETURN_TYPES = ("MESH",)
    INPUT_SOCKET_TYPES = {
        "mesh": "MESH",
        "base_color": "IMAGE",
        "metallic": "IMAGE",
        "roughness": "IMAGE",
        "occlusion": "IMAGE",
        "normal_map": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        mesh: tuple[str, int],
        base_color: tuple[str, int],
        metallic: tuple[str, int] | None = None,
        roughness: tuple[str, int] | None = None,
        occlusion: tuple[str, int] | None = None,
        normal_map: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mesh = mesh
        self.base_color = base_color
        if metallic is not None:
            self.metallic = metallic
        if roughness is not None:
            self.roughness = roughness
        if occlusion is not None:
            self.occlusion = occlusion
        if normal_map is not None:
            self.normal_map = normal_map


class BakeNormalMapFromMesh(BaseNode):
    __slots__ = (
        "low_poly",
        "high_poly",
        "resolution",
        "cage_distance",
        "ignore_backfaces",
    )
    CATEGORY = "3d/texturing"
    OUTPUTS = {"normal_map": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "low_poly": "MESH",
        "high_poly": "MESH",
        "resolution": "INT",
        "cage_distance": "FLOAT",
        "ignore_backfaces": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        low_poly: tuple[str, int],
        high_poly: tuple[str, int],
        resolution: int | tuple[str, int] = 1024,
        cage_distance: float | tuple[str, int] = 0.05,
        ignore_backfaces: bool | tuple[str, int] = True,
    ) -> None:
        super().__init__(node_id=node_id)
        self.low_poly = low_poly
        self.high_poly = high_poly
        self.resolution = resolution
        self.cage_distance = cage_distance
        self.ignore_backfaces = ignore_backfaces
        self.__validate()

    def __validate(self) -> None:
        validate_range("resolution", self.resolution, 64, 8192)
        validate_range("cage_distance", self.cage_distance, 0.001, 0.5)
        validate_bool("ignore_backfaces", self.ignore_backfaces)


class BakeAmbientOcclusion(BaseNode):
    __slots__ = (
        "low_poly",
        "high_poly",
        "resolution",
        "samples",
        "max_distance",
        "strength",
        "bias",
    )
    CATEGORY = "3d/texturing"
    OUTPUTS = {"occlusion": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "low_poly": "MESH",
        "high_poly": "MESH",
        "resolution": "INT",
        "samples": "INT",
        "max_distance": "FLOAT",
        "strength": "FLOAT",
        "bias": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        low_poly: tuple[str, int],
        high_poly: tuple[str, int],
        resolution: int | tuple[str, int] = 1024,
        samples: int | tuple[str, int] = 64,
        max_distance: float | tuple[str, int] = 0.5,
        strength: float | tuple[str, int] = 1.0,
        bias: float | tuple[str, int] = 0.01,
    ) -> None:
        super().__init__(node_id=node_id)
        self.low_poly = low_poly
        self.high_poly = high_poly
        self.resolution = resolution
        self.samples = samples
        self.max_distance = max_distance
        self.strength = strength
        self.bias = bias
        self.__validate()

    def __validate(self) -> None:
        validate_range("resolution", self.resolution, 64, 8192)
        validate_range("samples", self.samples, 4, 1024)
        validate_range("max_distance", self.max_distance, 0.01, 2.0)
        validate_range("strength", self.strength, 0.0, 2.0)
        validate_range("bias", self.bias, 0.0001, 0.2)
