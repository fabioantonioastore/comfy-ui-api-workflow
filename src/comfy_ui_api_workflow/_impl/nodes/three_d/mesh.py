from typing import Any, Literal, get_args

from ..base import BaseNode
from ..validation import validate_bool, validate_choice, validate_range
from ..dynamic import Combo, DynamicInputs, Flag, Number, validate_dynamic


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


DecimateMesh_Placement_Modes = Literal["midpoint", "qem"]


class DecimateMesh(DynamicInputs, BaseNode):
    __slots__ = (
        "mesh",
        "target_face_count",
        "placement_mode",
        "placement_mode_line_quadric_weight",
        "placement_mode_feature_edge_quadric_weight",
        "placement_mode_feature_edge_min_dihedral_deg",
        "placement_mode_clamp_v_to_edge",
    )
    CATEGORY = "3d/mesh"
    OUTPUTS = {"mesh": 0}
    RETURN_TYPES = ("MESH",)
    INPUT_NAMES = {
        "placement_mode_line_quadric_weight": "placement_mode.line_quadric_weight",
        "placement_mode_feature_edge_quadric_weight": "placement_mode.feature_edge_quadric_weight",
        "placement_mode_feature_edge_min_dihedral_deg": "placement_mode.feature_edge_min_dihedral_deg",
        "placement_mode_clamp_v_to_edge": "placement_mode.clamp_v_to_edge",
    }
    INPUT_SOCKET_TYPES = {
        "mesh": "MESH",
        "target_face_count": "INT",
        "placement_mode": "COMFY_DYNAMICCOMBO_V3",
        "placement_mode_line_quadric_weight": "FLOAT",
        "placement_mode_feature_edge_quadric_weight": "FLOAT",
        "placement_mode_feature_edge_min_dihedral_deg": "FLOAT",
        "placement_mode_clamp_v_to_edge": "BOOLEAN",
    }
    DYNAMIC_INPUTS = {
        "placement_mode": Combo(
            {
                "midpoint": {},
                "qem": {
                    "placement_mode_line_quadric_weight": Number(
                        default=0.0, low=0.0, high=100.0
                    ),
                    "placement_mode_feature_edge_quadric_weight": Number(
                        default=0.0, low=0.0, high=1000.0
                    ),
                    "placement_mode_feature_edge_min_dihedral_deg": Number(
                        default=30.0, low=0.0, high=180.0
                    ),
                    "placement_mode_clamp_v_to_edge": Flag(default=True),
                },
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        mesh: tuple[str, int],
        target_face_count: int | tuple[str, int] = 200000,
        placement_mode: DecimateMesh_Placement_Modes = "midpoint",
        placement_mode_line_quadric_weight: float | tuple[str, int] | None = None,
        placement_mode_feature_edge_quadric_weight: (
            float | tuple[str, int] | None
        ) = None,
        placement_mode_feature_edge_min_dihedral_deg: (
            float | tuple[str, int] | None
        ) = None,
        placement_mode_clamp_v_to_edge: bool | tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mesh = mesh
        self.target_face_count = target_face_count
        self.placement_mode = placement_mode
        if placement_mode_line_quadric_weight is not None:
            self.placement_mode_line_quadric_weight = placement_mode_line_quadric_weight
        if placement_mode_feature_edge_quadric_weight is not None:
            self.placement_mode_feature_edge_quadric_weight = (
                placement_mode_feature_edge_quadric_weight
            )
        if placement_mode_feature_edge_min_dihedral_deg is not None:
            self.placement_mode_feature_edge_min_dihedral_deg = (
                placement_mode_feature_edge_min_dihedral_deg
            )
        if placement_mode_clamp_v_to_edge is not None:
            self.placement_mode_clamp_v_to_edge = placement_mode_clamp_v_to_edge
        self.__validate()

    def __validate(self) -> None:
        validate_range("target_face_count", self.target_face_count, 0, 50000000)
        validate_dynamic(self)


class MergeMeshes(BaseNode):
    __slots__ = (
        "meshes_mesh0",
        "meshes_mesh1",
        "meshes_mesh2",
        "meshes_mesh3",
        "meshes_mesh4",
        "meshes_mesh5",
        "meshes_mesh6",
        "meshes_mesh7",
        "meshes_mesh8",
        "meshes_mesh9",
        "meshes_mesh10",
        "meshes_mesh11",
        "meshes_mesh12",
        "meshes_mesh13",
        "meshes_mesh14",
        "meshes_mesh15",
        "meshes_mesh16",
        "meshes_mesh17",
        "meshes_mesh18",
        "meshes_mesh19",
        "meshes_mesh20",
        "meshes_mesh21",
        "meshes_mesh22",
        "meshes_mesh23",
        "meshes_mesh24",
        "meshes_mesh25",
        "meshes_mesh26",
        "meshes_mesh27",
        "meshes_mesh28",
        "meshes_mesh29",
        "meshes_mesh30",
        "meshes_mesh31",
        "meshes_mesh32",
        "meshes_mesh33",
        "meshes_mesh34",
        "meshes_mesh35",
        "meshes_mesh36",
        "meshes_mesh37",
        "meshes_mesh38",
        "meshes_mesh39",
        "meshes_mesh40",
        "meshes_mesh41",
        "meshes_mesh42",
        "meshes_mesh43",
        "meshes_mesh44",
        "meshes_mesh45",
        "meshes_mesh46",
        "meshes_mesh47",
        "meshes_mesh48",
        "meshes_mesh49",
    )
    CATEGORY = "3d/mesh"
    OUTPUTS = {"mesh": 0}
    RETURN_TYPES = ("MESH",)
    INPUT_NAMES = {
        "meshes_mesh0": "meshes.mesh0",
        "meshes_mesh1": "meshes.mesh1",
        "meshes_mesh2": "meshes.mesh2",
        "meshes_mesh3": "meshes.mesh3",
        "meshes_mesh4": "meshes.mesh4",
        "meshes_mesh5": "meshes.mesh5",
        "meshes_mesh6": "meshes.mesh6",
        "meshes_mesh7": "meshes.mesh7",
        "meshes_mesh8": "meshes.mesh8",
        "meshes_mesh9": "meshes.mesh9",
        "meshes_mesh10": "meshes.mesh10",
        "meshes_mesh11": "meshes.mesh11",
        "meshes_mesh12": "meshes.mesh12",
        "meshes_mesh13": "meshes.mesh13",
        "meshes_mesh14": "meshes.mesh14",
        "meshes_mesh15": "meshes.mesh15",
        "meshes_mesh16": "meshes.mesh16",
        "meshes_mesh17": "meshes.mesh17",
        "meshes_mesh18": "meshes.mesh18",
        "meshes_mesh19": "meshes.mesh19",
        "meshes_mesh20": "meshes.mesh20",
        "meshes_mesh21": "meshes.mesh21",
        "meshes_mesh22": "meshes.mesh22",
        "meshes_mesh23": "meshes.mesh23",
        "meshes_mesh24": "meshes.mesh24",
        "meshes_mesh25": "meshes.mesh25",
        "meshes_mesh26": "meshes.mesh26",
        "meshes_mesh27": "meshes.mesh27",
        "meshes_mesh28": "meshes.mesh28",
        "meshes_mesh29": "meshes.mesh29",
        "meshes_mesh30": "meshes.mesh30",
        "meshes_mesh31": "meshes.mesh31",
        "meshes_mesh32": "meshes.mesh32",
        "meshes_mesh33": "meshes.mesh33",
        "meshes_mesh34": "meshes.mesh34",
        "meshes_mesh35": "meshes.mesh35",
        "meshes_mesh36": "meshes.mesh36",
        "meshes_mesh37": "meshes.mesh37",
        "meshes_mesh38": "meshes.mesh38",
        "meshes_mesh39": "meshes.mesh39",
        "meshes_mesh40": "meshes.mesh40",
        "meshes_mesh41": "meshes.mesh41",
        "meshes_mesh42": "meshes.mesh42",
        "meshes_mesh43": "meshes.mesh43",
        "meshes_mesh44": "meshes.mesh44",
        "meshes_mesh45": "meshes.mesh45",
        "meshes_mesh46": "meshes.mesh46",
        "meshes_mesh47": "meshes.mesh47",
        "meshes_mesh48": "meshes.mesh48",
        "meshes_mesh49": "meshes.mesh49",
    }
    INPUT_SOCKET_TYPES = {
        "meshes_mesh0": "MESH",
        "meshes_mesh1": "MESH",
        "meshes_mesh2": "MESH",
        "meshes_mesh3": "MESH",
        "meshes_mesh4": "MESH",
        "meshes_mesh5": "MESH",
        "meshes_mesh6": "MESH",
        "meshes_mesh7": "MESH",
        "meshes_mesh8": "MESH",
        "meshes_mesh9": "MESH",
        "meshes_mesh10": "MESH",
        "meshes_mesh11": "MESH",
        "meshes_mesh12": "MESH",
        "meshes_mesh13": "MESH",
        "meshes_mesh14": "MESH",
        "meshes_mesh15": "MESH",
        "meshes_mesh16": "MESH",
        "meshes_mesh17": "MESH",
        "meshes_mesh18": "MESH",
        "meshes_mesh19": "MESH",
        "meshes_mesh20": "MESH",
        "meshes_mesh21": "MESH",
        "meshes_mesh22": "MESH",
        "meshes_mesh23": "MESH",
        "meshes_mesh24": "MESH",
        "meshes_mesh25": "MESH",
        "meshes_mesh26": "MESH",
        "meshes_mesh27": "MESH",
        "meshes_mesh28": "MESH",
        "meshes_mesh29": "MESH",
        "meshes_mesh30": "MESH",
        "meshes_mesh31": "MESH",
        "meshes_mesh32": "MESH",
        "meshes_mesh33": "MESH",
        "meshes_mesh34": "MESH",
        "meshes_mesh35": "MESH",
        "meshes_mesh36": "MESH",
        "meshes_mesh37": "MESH",
        "meshes_mesh38": "MESH",
        "meshes_mesh39": "MESH",
        "meshes_mesh40": "MESH",
        "meshes_mesh41": "MESH",
        "meshes_mesh42": "MESH",
        "meshes_mesh43": "MESH",
        "meshes_mesh44": "MESH",
        "meshes_mesh45": "MESH",
        "meshes_mesh46": "MESH",
        "meshes_mesh47": "MESH",
        "meshes_mesh48": "MESH",
        "meshes_mesh49": "MESH",
    }

    def __init__(
        self,
        node_id: str,
        meshes_mesh0: tuple[str, int],
        meshes_mesh1: tuple[str, int],
        meshes_mesh2: tuple[str, int] | None = None,
        meshes_mesh3: tuple[str, int] | None = None,
        meshes_mesh4: tuple[str, int] | None = None,
        meshes_mesh5: tuple[str, int] | None = None,
        meshes_mesh6: tuple[str, int] | None = None,
        meshes_mesh7: tuple[str, int] | None = None,
        meshes_mesh8: tuple[str, int] | None = None,
        meshes_mesh9: tuple[str, int] | None = None,
        meshes_mesh10: tuple[str, int] | None = None,
        meshes_mesh11: tuple[str, int] | None = None,
        meshes_mesh12: tuple[str, int] | None = None,
        meshes_mesh13: tuple[str, int] | None = None,
        meshes_mesh14: tuple[str, int] | None = None,
        meshes_mesh15: tuple[str, int] | None = None,
        meshes_mesh16: tuple[str, int] | None = None,
        meshes_mesh17: tuple[str, int] | None = None,
        meshes_mesh18: tuple[str, int] | None = None,
        meshes_mesh19: tuple[str, int] | None = None,
        meshes_mesh20: tuple[str, int] | None = None,
        meshes_mesh21: tuple[str, int] | None = None,
        meshes_mesh22: tuple[str, int] | None = None,
        meshes_mesh23: tuple[str, int] | None = None,
        meshes_mesh24: tuple[str, int] | None = None,
        meshes_mesh25: tuple[str, int] | None = None,
        meshes_mesh26: tuple[str, int] | None = None,
        meshes_mesh27: tuple[str, int] | None = None,
        meshes_mesh28: tuple[str, int] | None = None,
        meshes_mesh29: tuple[str, int] | None = None,
        meshes_mesh30: tuple[str, int] | None = None,
        meshes_mesh31: tuple[str, int] | None = None,
        meshes_mesh32: tuple[str, int] | None = None,
        meshes_mesh33: tuple[str, int] | None = None,
        meshes_mesh34: tuple[str, int] | None = None,
        meshes_mesh35: tuple[str, int] | None = None,
        meshes_mesh36: tuple[str, int] | None = None,
        meshes_mesh37: tuple[str, int] | None = None,
        meshes_mesh38: tuple[str, int] | None = None,
        meshes_mesh39: tuple[str, int] | None = None,
        meshes_mesh40: tuple[str, int] | None = None,
        meshes_mesh41: tuple[str, int] | None = None,
        meshes_mesh42: tuple[str, int] | None = None,
        meshes_mesh43: tuple[str, int] | None = None,
        meshes_mesh44: tuple[str, int] | None = None,
        meshes_mesh45: tuple[str, int] | None = None,
        meshes_mesh46: tuple[str, int] | None = None,
        meshes_mesh47: tuple[str, int] | None = None,
        meshes_mesh48: tuple[str, int] | None = None,
        meshes_mesh49: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.meshes_mesh0 = meshes_mesh0
        self.meshes_mesh1 = meshes_mesh1
        if meshes_mesh2 is not None:
            self.meshes_mesh2 = meshes_mesh2
        if meshes_mesh3 is not None:
            self.meshes_mesh3 = meshes_mesh3
        if meshes_mesh4 is not None:
            self.meshes_mesh4 = meshes_mesh4
        if meshes_mesh5 is not None:
            self.meshes_mesh5 = meshes_mesh5
        if meshes_mesh6 is not None:
            self.meshes_mesh6 = meshes_mesh6
        if meshes_mesh7 is not None:
            self.meshes_mesh7 = meshes_mesh7
        if meshes_mesh8 is not None:
            self.meshes_mesh8 = meshes_mesh8
        if meshes_mesh9 is not None:
            self.meshes_mesh9 = meshes_mesh9
        if meshes_mesh10 is not None:
            self.meshes_mesh10 = meshes_mesh10
        if meshes_mesh11 is not None:
            self.meshes_mesh11 = meshes_mesh11
        if meshes_mesh12 is not None:
            self.meshes_mesh12 = meshes_mesh12
        if meshes_mesh13 is not None:
            self.meshes_mesh13 = meshes_mesh13
        if meshes_mesh14 is not None:
            self.meshes_mesh14 = meshes_mesh14
        if meshes_mesh15 is not None:
            self.meshes_mesh15 = meshes_mesh15
        if meshes_mesh16 is not None:
            self.meshes_mesh16 = meshes_mesh16
        if meshes_mesh17 is not None:
            self.meshes_mesh17 = meshes_mesh17
        if meshes_mesh18 is not None:
            self.meshes_mesh18 = meshes_mesh18
        if meshes_mesh19 is not None:
            self.meshes_mesh19 = meshes_mesh19
        if meshes_mesh20 is not None:
            self.meshes_mesh20 = meshes_mesh20
        if meshes_mesh21 is not None:
            self.meshes_mesh21 = meshes_mesh21
        if meshes_mesh22 is not None:
            self.meshes_mesh22 = meshes_mesh22
        if meshes_mesh23 is not None:
            self.meshes_mesh23 = meshes_mesh23
        if meshes_mesh24 is not None:
            self.meshes_mesh24 = meshes_mesh24
        if meshes_mesh25 is not None:
            self.meshes_mesh25 = meshes_mesh25
        if meshes_mesh26 is not None:
            self.meshes_mesh26 = meshes_mesh26
        if meshes_mesh27 is not None:
            self.meshes_mesh27 = meshes_mesh27
        if meshes_mesh28 is not None:
            self.meshes_mesh28 = meshes_mesh28
        if meshes_mesh29 is not None:
            self.meshes_mesh29 = meshes_mesh29
        if meshes_mesh30 is not None:
            self.meshes_mesh30 = meshes_mesh30
        if meshes_mesh31 is not None:
            self.meshes_mesh31 = meshes_mesh31
        if meshes_mesh32 is not None:
            self.meshes_mesh32 = meshes_mesh32
        if meshes_mesh33 is not None:
            self.meshes_mesh33 = meshes_mesh33
        if meshes_mesh34 is not None:
            self.meshes_mesh34 = meshes_mesh34
        if meshes_mesh35 is not None:
            self.meshes_mesh35 = meshes_mesh35
        if meshes_mesh36 is not None:
            self.meshes_mesh36 = meshes_mesh36
        if meshes_mesh37 is not None:
            self.meshes_mesh37 = meshes_mesh37
        if meshes_mesh38 is not None:
            self.meshes_mesh38 = meshes_mesh38
        if meshes_mesh39 is not None:
            self.meshes_mesh39 = meshes_mesh39
        if meshes_mesh40 is not None:
            self.meshes_mesh40 = meshes_mesh40
        if meshes_mesh41 is not None:
            self.meshes_mesh41 = meshes_mesh41
        if meshes_mesh42 is not None:
            self.meshes_mesh42 = meshes_mesh42
        if meshes_mesh43 is not None:
            self.meshes_mesh43 = meshes_mesh43
        if meshes_mesh44 is not None:
            self.meshes_mesh44 = meshes_mesh44
        if meshes_mesh45 is not None:
            self.meshes_mesh45 = meshes_mesh45
        if meshes_mesh46 is not None:
            self.meshes_mesh46 = meshes_mesh46
        if meshes_mesh47 is not None:
            self.meshes_mesh47 = meshes_mesh47
        if meshes_mesh48 is not None:
            self.meshes_mesh48 = meshes_mesh48
        if meshes_mesh49 is not None:
            self.meshes_mesh49 = meshes_mesh49


RemeshMesh_Sign_Modes = Literal["udf", "sdf"]


class RemeshMesh(DynamicInputs, BaseNode):
    __slots__ = (
        "mesh",
        "resolution",
        "sign_mode",
        "sign_mode_qef",
        "sign_mode_drop_inverted_components",
        "sign_mode_drop_enclosed_components",
        "sign_mode_manifold",
        "band",
        "project_back",
        "fix_poles",
        "smooth_iters",
        "drop_small_components",
        "precluster_max_verts",
    )
    CATEGORY = "3d/mesh"
    OUTPUTS = {"mesh": 0}
    RETURN_TYPES = ("MESH",)
    INPUT_NAMES = {
        "sign_mode_qef": "sign_mode.qef",
        "sign_mode_drop_inverted_components": "sign_mode.drop_inverted_components",
        "sign_mode_drop_enclosed_components": "sign_mode.drop_enclosed_components",
        "sign_mode_manifold": "sign_mode.manifold",
    }
    INPUT_SOCKET_TYPES = {
        "mesh": "MESH",
        "resolution": "INT",
        "sign_mode": "COMFY_DYNAMICCOMBO_V3",
        "sign_mode_qef": "BOOLEAN",
        "sign_mode_drop_inverted_components": "BOOLEAN",
        "sign_mode_drop_enclosed_components": "BOOLEAN",
        "sign_mode_manifold": "BOOLEAN",
        "band": "FLOAT",
        "project_back": "FLOAT",
        "fix_poles": "BOOLEAN",
        "smooth_iters": "INT",
        "drop_small_components": "FLOAT",
        "precluster_max_verts": "INT",
    }
    DYNAMIC_INPUTS = {
        "sign_mode": Combo(
            {
                "udf": {
                    "sign_mode_qef": Flag(default=False),
                    "sign_mode_drop_inverted_components": Flag(default=False),
                    "sign_mode_drop_enclosed_components": Flag(default=False),
                },
                "sdf": {
                    "sign_mode_qef": Flag(default=True),
                    "sign_mode_manifold": Flag(default=False),
                },
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        mesh: tuple[str, int],
        resolution: int | tuple[str, int] = 512,
        sign_mode: RemeshMesh_Sign_Modes = "udf",
        sign_mode_qef: bool | tuple[str, int] | None = None,
        sign_mode_drop_inverted_components: bool | tuple[str, int] | None = None,
        sign_mode_drop_enclosed_components: bool | tuple[str, int] | None = None,
        sign_mode_manifold: bool | tuple[str, int] | None = None,
        band: float | tuple[str, int] = 1.0,
        project_back: float | tuple[str, int] = 0.0,
        fix_poles: bool | tuple[str, int] = False,
        smooth_iters: int | tuple[str, int] = 0,
        drop_small_components: float | tuple[str, int] = 0.01,
        precluster_max_verts: int | tuple[str, int] = 20000000,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mesh = mesh
        self.resolution = resolution
        self.sign_mode = sign_mode
        if sign_mode_qef is not None:
            self.sign_mode_qef = sign_mode_qef
        if sign_mode_drop_inverted_components is not None:
            self.sign_mode_drop_inverted_components = sign_mode_drop_inverted_components
        if sign_mode_drop_enclosed_components is not None:
            self.sign_mode_drop_enclosed_components = sign_mode_drop_enclosed_components
        if sign_mode_manifold is not None:
            self.sign_mode_manifold = sign_mode_manifold
        self.band = band
        self.project_back = project_back
        self.fix_poles = fix_poles
        self.smooth_iters = smooth_iters
        self.drop_small_components = drop_small_components
        self.precluster_max_verts = precluster_max_verts
        self.__validate()

    def __validate(self) -> None:
        validate_range("resolution", self.resolution, 32, 2048)
        validate_range("band", self.band, 0.5, 4.0)
        validate_range("project_back", self.project_back, 0.0, 1.0)
        validate_bool("fix_poles", self.fix_poles)
        validate_range("smooth_iters", self.smooth_iters, 0, 20)
        validate_range("drop_small_components", self.drop_small_components, 0.0, 0.5)
        validate_range("precluster_max_verts", self.precluster_max_verts, 0, 100000000)
        validate_dynamic(self)


RotateMesh_Modes = Literal["euler_xyz", "quaternion"]


class RotateMesh(DynamicInputs, BaseNode):
    __slots__ = (
        "mesh",
        "mode",
        "mode_angle_x",
        "mode_angle_y",
        "mode_angle_z",
        "mode_qw",
        "mode_qx",
        "mode_qy",
        "mode_qz",
    )
    CATEGORY = "3d/mesh"
    OUTPUTS = {"mesh": 0}
    RETURN_TYPES = ("MESH",)
    INPUT_NAMES = {
        "mode_angle_x": "mode.angle_x",
        "mode_angle_y": "mode.angle_y",
        "mode_angle_z": "mode.angle_z",
        "mode_qw": "mode.qw",
        "mode_qx": "mode.qx",
        "mode_qy": "mode.qy",
        "mode_qz": "mode.qz",
    }
    INPUT_SOCKET_TYPES = {
        "mesh": "MESH",
        "mode": "COMFY_DYNAMICCOMBO_V3",
        "mode_angle_x": "FLOAT",
        "mode_angle_y": "FLOAT",
        "mode_angle_z": "FLOAT",
        "mode_qw": "FLOAT",
        "mode_qx": "FLOAT",
        "mode_qy": "FLOAT",
        "mode_qz": "FLOAT",
    }
    DYNAMIC_INPUTS = {
        "mode": Combo(
            {
                "euler_xyz": {
                    "mode_angle_x": Number(default=0.0, low=-360.0, high=360.0),
                    "mode_angle_y": Number(default=0.0, low=-360.0, high=360.0),
                    "mode_angle_z": Number(default=0.0, low=-360.0, high=360.0),
                },
                "quaternion": {
                    "mode_qw": Number(default=1.0, low=-1.0, high=1.0),
                    "mode_qx": Number(default=0.0, low=-1.0, high=1.0),
                    "mode_qy": Number(default=0.0, low=-1.0, high=1.0),
                    "mode_qz": Number(default=0.0, low=-1.0, high=1.0),
                },
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        mesh: tuple[str, int],
        mode: RotateMesh_Modes = "euler_xyz",
        mode_angle_x: float | tuple[str, int] | None = None,
        mode_angle_y: float | tuple[str, int] | None = None,
        mode_angle_z: float | tuple[str, int] | None = None,
        mode_qw: float | tuple[str, int] | None = None,
        mode_qx: float | tuple[str, int] | None = None,
        mode_qy: float | tuple[str, int] | None = None,
        mode_qz: float | tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mesh = mesh
        self.mode = mode
        if mode_angle_x is not None:
            self.mode_angle_x = mode_angle_x
        if mode_angle_y is not None:
            self.mode_angle_y = mode_angle_y
        if mode_angle_z is not None:
            self.mode_angle_z = mode_angle_z
        if mode_qw is not None:
            self.mode_qw = mode_qw
        if mode_qx is not None:
            self.mode_qx = mode_qx
        if mode_qy is not None:
            self.mode_qy = mode_qy
        if mode_qz is not None:
            self.mode_qz = mode_qz
        self.__validate()

    def __validate(self) -> None:
        validate_dynamic(self)
