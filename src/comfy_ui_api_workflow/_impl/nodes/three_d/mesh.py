from typing import Any, Literal, get_args

from ..base import BaseNode
from ..validation import validate_bool, validate_choice, validate_range


class FillHoles(BaseNode):
    __slots__ = (
        "mesh",
        "max_perimeter",
        "weld_epsilon_rel",
        "max_vertices",
        "fill_chains",
    )
    CATEGORY = "3d/mesh"
    OUTPUTS = {"mesh": 0}
    RETURN_TYPES = ("MESH",)
    INPUT_SOCKET_TYPES = {
        "mesh": "MESH",
        "max_perimeter": "FLOAT",
        "weld_epsilon_rel": "FLOAT",
        "max_vertices": "INT",
        "fill_chains": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        mesh: tuple[str, int],
        max_perimeter: float | tuple[str, int] = 0.03,
        weld_epsilon_rel: float | tuple[str, int] = 1e-05,
        max_vertices: int | tuple[str, int] = 16,
        fill_chains: bool | tuple[str, int] = False,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mesh = mesh
        self.max_perimeter = max_perimeter
        self.weld_epsilon_rel = weld_epsilon_rel
        self.max_vertices = max_vertices
        self.fill_chains = fill_chains
        self.__validate()

    def __validate(self) -> None:
        validate_range("max_perimeter", self.max_perimeter, 0.0)
        validate_range("weld_epsilon_rel", self.weld_epsilon_rel, 0.0)
        validate_range("max_vertices", self.max_vertices, 3, 1024)
        validate_bool("fill_chains", self.fill_chains)


class WeldVertices(BaseNode):
    __slots__ = ("mesh", "epsilon_rel", "epsilon_abs")
    CATEGORY = "3d/mesh"
    OUTPUTS = {"mesh": 0}
    RETURN_TYPES = ("MESH",)
    INPUT_SOCKET_TYPES = {
        "mesh": "MESH",
        "epsilon_rel": "FLOAT",
        "epsilon_abs": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        mesh: tuple[str, int],
        epsilon_rel: float | tuple[str, int] = 1e-05,
        epsilon_abs: float | tuple[str, int] = 0.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mesh = mesh
        self.epsilon_rel = epsilon_rel
        self.epsilon_abs = epsilon_abs
        self.__validate()

    def __validate(self) -> None:
        validate_range("epsilon_rel", self.epsilon_rel, 0.0)
        validate_range("epsilon_abs", self.epsilon_abs, 0.0)


class PaintMesh(BaseNode):
    __slots__ = ("mesh", "voxel_colors")
    CATEGORY = "3d/mesh"
    OUTPUTS = {"mesh": 0}
    RETURN_TYPES = ("MESH",)
    INPUT_SOCKET_TYPES = {"mesh": "MESH", "voxel_colors": "VOXEL"}

    def __init__(
        self, node_id: str, mesh: tuple[str, int], voxel_colors: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.mesh = mesh
        self.voxel_colors = voxel_colors


RenderMesh_Modes = Literal[
    "auto", "texture", "vertex colors", "solid", "normal", "depth"
]
RENDERMESH_MODES = frozenset(get_args(RenderMesh_Modes))


class RenderMesh(BaseNode):
    __slots__ = (
        "mesh",
        "mode",
        "width",
        "height",
        "background",
        "model_3d_info",
        "camera_info",
    )
    CATEGORY = "3d/mesh"
    OUTPUTS = {"image": 0, "mask": 1}
    RETURN_TYPES = ("IMAGE", "MASK")
    INPUT_SOCKET_TYPES = {
        "mesh": "MESH",
        "mode": "COMBO",
        "width": "INT",
        "height": "INT",
        "background": "COLOR",
        "model_3d_info": "LOAD3D_MODEL_INFO",
        "camera_info": "LOAD3D_CAMERA",
    }

    def __init__(
        self,
        node_id: str,
        mesh: tuple[str, int],
        mode: RenderMesh_Modes | tuple[str, int],
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
        background: Any = "#000000",
        model_3d_info: tuple[str, int] | None = None,
        camera_info: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mesh = mesh
        self.mode = mode
        self.width = width
        self.height = height
        self.background = background
        if model_3d_info is not None:
            self.model_3d_info = model_3d_info
        if camera_info is not None:
            self.camera_info = camera_info
        self.__validate()

    def __validate(self) -> None:
        validate_choice("mode", self.mode, RENDERMESH_MODES)
        validate_range("width", self.width, 64, 4096)
        validate_range("height", self.height, 64, 4096)


class MeshSmoothNormals(BaseNode):
    __slots__ = ("mesh", "crease_angle")
    CATEGORY = "3d/mesh"
    OUTPUTS = {"mesh": 0}
    RETURN_TYPES = ("MESH",)
    INPUT_SOCKET_TYPES = {"mesh": "MESH", "crease_angle": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        mesh: tuple[str, int],
        crease_angle: float | tuple[str, int] = 180.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mesh = mesh
        self.crease_angle = crease_angle
        self.__validate()

    def __validate(self) -> None:
        validate_range("crease_angle", self.crease_angle, 0.0, 180.0)


class GetMeshInfo(BaseNode):
    __slots__ = ("mesh",)
    CATEGORY = "3d/mesh"
    OUTPUTS = {"mesh": 0, "info": 1}
    RETURN_TYPES = ("MESH", "STRING")
    INPUT_SOCKET_TYPES = {"mesh": "MESH"}

    def __init__(self, node_id: str, mesh: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.mesh = mesh
