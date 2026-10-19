from typing import Any, Literal, get_args

from ..base import BaseNode
from ..validation import validate_choice, validate_not_empty, validate_range
from ...lazy import submodules_of
from ..dynamic import Choice, Combo, DynamicInputs, Flag, Number, validate_dynamic


class Load3D(BaseNode):
    __slots__ = ("model_file", "image", "width", "height")
    CATEGORY = "3d"
    OUTPUTS = {
        "image": 0,
        "mask": 1,
        "mesh_path": 2,
        "normal": 3,
        "camera_info": 4,
        "recording_video": 5,
        "model_3d": 6,
        "model_3d_info": 7,
    }
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "STRING",
        "IMAGE",
        "LOAD3D_CAMERA",
        "VIDEO",
        "FILE_3D",
        "LOAD3D_MODEL_INFO",
    )
    INPUT_SOCKET_TYPES = {
        "model_file": "COMBO",
        "image": "LOAD_3D",
        "width": "INT",
        "height": "INT",
    }

    def __init__(
        self,
        node_id: str,
        model_file: str | tuple[str, int],
        image: Any,
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model_file = model_file
        self.image = image
        self.width = width
        self.height = height
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("model_file", self.model_file)
        validate_range("width", self.width, 1, 4096)
        validate_range("height", self.height, 1, 4096)


class Load3DAdvanced(BaseNode):
    __slots__ = ("model_file", "viewport_state", "width", "height")
    CATEGORY = "3d"
    OUTPUTS = {
        "model_3d": 0,
        "model_3d_info": 1,
        "camera_info": 2,
        "width": 3,
        "height": 4,
    }
    RETURN_TYPES = ("FILE_3D", "LOAD3D_MODEL_INFO", "LOAD3D_CAMERA", "INT", "INT")
    INPUT_SOCKET_TYPES = {
        "model_file": "COMBO",
        "viewport_state": "LOAD_3D",
        "width": "INT",
        "height": "INT",
    }

    def __init__(
        self,
        node_id: str,
        model_file: str | tuple[str, int],
        viewport_state: Any,
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model_file = model_file
        self.viewport_state = viewport_state
        self.width = width
        self.height = height
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("model_file", self.model_file)
        validate_range("width", self.width, 1, 4096)
        validate_range("height", self.height, 1, 4096)


class Preview3DAdvanced(BaseNode):
    __slots__ = (
        "model_3d",
        "viewport_state",
        "width",
        "height",
        "model_3d_info",
        "camera_info",
    )
    CATEGORY = "3d"
    OUTPUTS = {
        "model_3d": 0,
        "model_3d_info": 1,
        "camera_info": 2,
        "width": 3,
        "height": 4,
    }
    RETURN_TYPES = ("FILE_3D", "LOAD3D_MODEL_INFO", "LOAD3D_CAMERA", "INT", "INT")
    INPUT_SOCKET_TYPES = {
        "model_3d": "FILE_3D_GLB,FILE_3D_GLTF,FILE_3D_FBX,FILE_3D_OBJ,FILE_3D_STL,FILE_3D_USDZ,FILE_3D",
        "viewport_state": "LOAD_3D",
        "width": "INT",
        "height": "INT",
        "model_3d_info": "LOAD3D_MODEL_INFO",
        "camera_info": "LOAD3D_CAMERA",
    }

    def __init__(
        self,
        node_id: str,
        model_3d: tuple[str, int],
        viewport_state: Any,
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
        model_3d_info: tuple[str, int] | None = None,
        camera_info: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model_3d = model_3d
        self.viewport_state = viewport_state
        self.width = width
        self.height = height
        if model_3d_info is not None:
            self.model_3d_info = model_3d_info
        if camera_info is not None:
            self.camera_info = camera_info
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 1, 4096)
        validate_range("height", self.height, 1, 4096)


class PreviewGaussianSplat(BaseNode):
    __slots__ = (
        "model_3d",
        "viewport_state",
        "width",
        "height",
        "model_3d_info",
        "camera_info",
    )
    CATEGORY = "3d"
    OUTPUTS = {
        "model_3d": 0,
        "model_3d_info": 1,
        "camera_info": 2,
        "width": 3,
        "height": 4,
    }
    RETURN_TYPES = (
        "FILE_3D_SPLAT_ANY",
        "LOAD3D_MODEL_INFO",
        "LOAD3D_CAMERA",
        "INT",
        "INT",
    )
    INPUT_SOCKET_TYPES = {
        "model_3d": "FILE_3D_SPLAT_ANY,FILE_3D_PLY,FILE_3D_SPLAT,FILE_3D_SPZ,FILE_3D_KSPLAT",
        "viewport_state": "LOAD_3D",
        "width": "INT",
        "height": "INT",
        "model_3d_info": "LOAD3D_MODEL_INFO",
        "camera_info": "LOAD3D_CAMERA",
    }

    def __init__(
        self,
        node_id: str,
        model_3d: tuple[str, int],
        viewport_state: Any,
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
        model_3d_info: tuple[str, int] | None = None,
        camera_info: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model_3d = model_3d
        self.viewport_state = viewport_state
        self.width = width
        self.height = height
        if model_3d_info is not None:
            self.model_3d_info = model_3d_info
        if camera_info is not None:
            self.camera_info = camera_info
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 1, 4096)
        validate_range("height", self.height, 1, 4096)


class PreviewPointCloud(BaseNode):
    __slots__ = (
        "model_3d",
        "viewport_state",
        "width",
        "height",
        "model_3d_info",
        "camera_info",
    )
    CATEGORY = "3d"
    OUTPUTS = {
        "model_3d": 0,
        "model_3d_info": 1,
        "camera_info": 2,
        "width": 3,
        "height": 4,
    }
    RETURN_TYPES = (
        "FILE_3D_POINT_CLOUD_ANY",
        "LOAD3D_MODEL_INFO",
        "LOAD3D_CAMERA",
        "INT",
        "INT",
    )
    INPUT_SOCKET_TYPES = {
        "model_3d": "FILE_3D_POINT_CLOUD_ANY,FILE_3D_PLY",
        "viewport_state": "LOAD_3D",
        "width": "INT",
        "height": "INT",
        "model_3d_info": "LOAD3D_MODEL_INFO",
        "camera_info": "LOAD3D_CAMERA",
    }

    def __init__(
        self,
        node_id: str,
        model_3d: tuple[str, int],
        viewport_state: Any,
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
        model_3d_info: tuple[str, int] | None = None,
        camera_info: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model_3d = model_3d
        self.viewport_state = viewport_state
        self.width = width
        self.height = height
        if model_3d_info is not None:
            self.model_3d_info = model_3d_info
        if camera_info is not None:
            self.camera_info = camera_info
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 1, 4096)
        validate_range("height", self.height, 1, 4096)


VoxelToMesh_Algorithms = Literal["surface net", "basic"]
VOXELTOMESH_ALGORITHMS = frozenset(get_args(VoxelToMesh_Algorithms))


class VoxelToMesh(BaseNode):
    __slots__ = ("voxel", "algorithm", "threshold")
    CATEGORY = "3d"
    OUTPUTS = {"MESH": 0}
    RETURN_TYPES = ("MESH",)
    INPUT_SOCKET_TYPES = {"voxel": "VOXEL", "algorithm": "COMBO", "threshold": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        voxel: tuple[str, int],
        algorithm: VoxelToMesh_Algorithms | tuple[str, int],
        threshold: float | tuple[str, int] = 0.6,
    ) -> None:
        super().__init__(node_id=node_id)
        self.voxel = voxel
        self.algorithm = algorithm
        self.threshold = threshold
        self.__validate()

    def __validate(self) -> None:
        validate_choice("algorithm", self.algorithm, VOXELTOMESH_ALGORITHMS)
        validate_range("threshold", self.threshold, -1.0, 1.0)


class SaveGLB(BaseNode):
    __slots__ = ("mesh", "filename_prefix")
    CATEGORY = "3d"
    INPUT_SOCKET_TYPES = {
        "mesh": "MESH,FILE_3D_GLB,FILE_3D_GLTF,FILE_3D_OBJ,FILE_3D_FBX,FILE_3D_STL,FILE_3D_USDZ,FILE_3D_PLY,FILE_3D_SPLAT,FILE_3D_SPZ,FILE_3D_KSPLAT,FILE_3D_SPLAT_ANY,FILE_3D_POINT_CLOUD_ANY,FILE_3D",
        "filename_prefix": "STRING",
    }

    def __init__(
        self,
        node_id: str,
        mesh: tuple[str, int],
        filename_prefix: str | tuple[str, int] = "3d/ComfyUI",
    ) -> None:
        super().__init__(node_id=node_id)
        self.mesh = mesh
        self.filename_prefix = filename_prefix


class MeshToFile3D(BaseNode):
    __slots__ = ("mesh",)
    CATEGORY = "3d"
    OUTPUTS = {"model_3d": 0}
    RETURN_TYPES = ("FILE_3D_GLB",)
    INPUT_SOCKET_TYPES = {"mesh": "MESH"}

    def __init__(self, node_id: str, mesh: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.mesh = mesh


class Save3DAdvanced(BaseNode):
    __slots__ = (
        "model_3d",
        "viewport_state",
        "filename_prefix",
        "width",
        "height",
        "model_3d_info",
        "camera_info",
    )
    CATEGORY = "3d"
    OUTPUTS = {
        "model_3d": 0,
        "model_3d_info": 1,
        "camera_info": 2,
        "width": 3,
        "height": 4,
    }
    RETURN_TYPES = ("FILE_3D", "LOAD3D_MODEL_INFO", "LOAD3D_CAMERA", "INT", "INT")
    INPUT_SOCKET_TYPES = {
        "model_3d": "FILE_3D_GLB,FILE_3D_GLTF,FILE_3D_FBX,FILE_3D_OBJ,FILE_3D_STL,FILE_3D_USDZ,FILE_3D",
        "filename_prefix": "STRING",
        "viewport_state": "LOAD_3D",
        "width": "INT",
        "height": "INT",
        "model_3d_info": "LOAD3D_MODEL_INFO",
        "camera_info": "LOAD3D_CAMERA",
    }

    def __init__(
        self,
        node_id: str,
        model_3d: tuple[str, int],
        viewport_state: Any,
        filename_prefix: str | tuple[str, int] = "3d/ComfyUI",
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
        model_3d_info: tuple[str, int] | None = None,
        camera_info: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model_3d = model_3d
        self.viewport_state = viewport_state
        self.filename_prefix = filename_prefix
        self.width = width
        self.height = height
        if model_3d_info is not None:
            self.model_3d_info = model_3d_info
        if camera_info is not None:
            self.camera_info = camera_info
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 1, 4096)
        validate_range("height", self.height, 1, 4096)


class SaveGaussianSplat(BaseNode):
    __slots__ = (
        "model_3d",
        "viewport_state",
        "filename_prefix",
        "width",
        "height",
        "model_3d_info",
        "camera_info",
    )
    CATEGORY = "3d"
    OUTPUTS = {
        "model_3d": 0,
        "model_3d_info": 1,
        "camera_info": 2,
        "width": 3,
        "height": 4,
    }
    RETURN_TYPES = (
        "FILE_3D_SPLAT_ANY",
        "LOAD3D_MODEL_INFO",
        "LOAD3D_CAMERA",
        "INT",
        "INT",
    )
    INPUT_SOCKET_TYPES = {
        "model_3d": "FILE_3D_SPLAT_ANY,FILE_3D_PLY,FILE_3D_SPLAT,FILE_3D_SPZ,FILE_3D_KSPLAT",
        "filename_prefix": "STRING",
        "viewport_state": "LOAD_3D",
        "width": "INT",
        "height": "INT",
        "model_3d_info": "LOAD3D_MODEL_INFO",
        "camera_info": "LOAD3D_CAMERA",
    }

    def __init__(
        self,
        node_id: str,
        model_3d: tuple[str, int],
        viewport_state: Any,
        filename_prefix: str | tuple[str, int] = "3d/ComfyUI",
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
        model_3d_info: tuple[str, int] | None = None,
        camera_info: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model_3d = model_3d
        self.viewport_state = viewport_state
        self.filename_prefix = filename_prefix
        self.width = width
        self.height = height
        if model_3d_info is not None:
            self.model_3d_info = model_3d_info
        if camera_info is not None:
            self.camera_info = camera_info
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 1, 4096)
        validate_range("height", self.height, 1, 4096)


class SavePointCloud(BaseNode):
    __slots__ = (
        "model_3d",
        "viewport_state",
        "filename_prefix",
        "width",
        "height",
        "model_3d_info",
        "camera_info",
    )
    CATEGORY = "3d"
    OUTPUTS = {
        "model_3d": 0,
        "model_3d_info": 1,
        "camera_info": 2,
        "width": 3,
        "height": 4,
    }
    RETURN_TYPES = (
        "FILE_3D_POINT_CLOUD_ANY",
        "LOAD3D_MODEL_INFO",
        "LOAD3D_CAMERA",
        "INT",
        "INT",
    )
    INPUT_SOCKET_TYPES = {
        "model_3d": "FILE_3D_POINT_CLOUD_ANY,FILE_3D_PLY",
        "filename_prefix": "STRING",
        "viewport_state": "LOAD_3D",
        "width": "INT",
        "height": "INT",
        "model_3d_info": "LOAD3D_MODEL_INFO",
        "camera_info": "LOAD3D_CAMERA",
    }

    def __init__(
        self,
        node_id: str,
        model_3d: tuple[str, int],
        viewport_state: Any,
        filename_prefix: str | tuple[str, int] = "3d/ComfyUI",
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
        model_3d_info: tuple[str, int] | None = None,
        camera_info: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model_3d = model_3d
        self.viewport_state = viewport_state
        self.filename_prefix = filename_prefix
        self.width = width
        self.height = height
        if model_3d_info is not None:
            self.model_3d_info = model_3d_info
        if camera_info is not None:
            self.camera_info = camera_info
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 1, 4096)
        validate_range("height", self.height, 1, 4096)


class Get3DComponents(BaseNode):
    __slots__ = ("model_3d",)
    CATEGORY = "3d"
    OUTPUTS = {"mesh": 0}
    RETURN_TYPES = ("MESH",)
    INPUT_SOCKET_TYPES = {
        "model_3d": "FILE_3D_GLB,FILE_3D_GLTF,FILE_3D_OBJ,FILE_3D_STL,FILE_3D"
    }

    def __init__(self, node_id: str, model_3d: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.model_3d = model_3d


class CameraAngle(BaseNode):
    __slots__ = ("horizontal_angle", "vertical_angle", "zoom", "image", "view")
    CATEGORY = "3d"
    OUTPUTS = {"camera_info": 0, "prompt": 1}
    RETURN_TYPES = ("LOAD3D_CAMERA", "STRING")
    INPUT_SOCKET_TYPES = {
        "horizontal_angle": "INT",
        "vertical_angle": "INT",
        "zoom": "FLOAT",
        "image": "IMAGE",
        "view": "STRING",
    }

    def __init__(
        self,
        node_id: str,
        horizontal_angle: int | tuple[str, int] = 0,
        vertical_angle: int | tuple[str, int] = 0,
        zoom: float | tuple[str, int] = 5.0,
        image: tuple[str, int] | None = None,
        view: Any = "",
    ) -> None:
        super().__init__(node_id=node_id)
        self.horizontal_angle = horizontal_angle
        self.vertical_angle = vertical_angle
        self.zoom = zoom
        if image is not None:
            self.image = image
        self.view = view
        self.__validate()

    def __validate(self) -> None:
        validate_range("horizontal_angle", self.horizontal_angle, 0, 360)
        validate_range("vertical_angle", self.vertical_angle, -30, 60)
        validate_range("zoom", self.zoom, 0.0, 10.0)


BuildPoseFile_Formats = Literal["glb", "bvh"]


BuildPoseFile_Camera_Translations = Literal["off", "centered", "absolute"]


BUILDPOSEFILE_CAMERA_TRANSLATIONS = frozenset(
    get_args(BuildPoseFile_Camera_Translations)
)


BuildPoseFile_Format_Mesh_Styles = Literal[
    "body_mesh", "bones_only", "openpose", "scail"
]


BuildPoseFile_Format_Mesh_Style_Bone_Vis = Literal["off", "octahedrons"]


BuildPoseFile_Format_Mesh_Style_Bone_Vis_Bone_Vis_Colors = Literal["white", "rainbow_y"]


BuildPoseFile_Format_Mesh_Style_Shaders = Literal[
    "default", "rainbow", "rainbow_face_normal", "rainbow_face_semantic"
]


BuildPoseFile_Format_Mesh_Style_Face_Styles = Literal["disabled", "full", "eyes_mouth"]


BuildPoseFile_Format_Units = Literal["cm", "m"]


class BuildPoseFile(DynamicInputs, BaseNode):
    __slots__ = (
        "pose_data",
        "format",
        "format_mesh_style",
        "format_mesh_style_bone_vis",
        "format_mesh_style_bone_vis_bone_vis_radius_m",
        "format_mesh_style_bone_vis_bone_vis_color",
        "format_mesh_style_shader",
        "format_mesh_style_shader_rainbow_tilt_z",
        "format_mesh_style_shader_rainbow_tilt_x",
        "format_mesh_style_shader_person_palette_falloff",
        "format_mesh_style_marker_radius_m",
        "format_mesh_style_stick_radius_m",
        "format_mesh_style_include_hands",
        "format_mesh_style_hand_marker_radius_m",
        "format_mesh_style_hand_stick_radius_m",
        "format_mesh_style_face_style",
        "format_mesh_style_face_marker_radius_m",
        "format_mesh_style_material_roughness",
        "format_bone_smooth_window",
        "format_units",
        "fps",
        "camera_translation",
        "track_index",
        "sam3d_body_model",
    )
    CATEGORY = "3d"
    OUTPUTS = {"model_3d": 0}
    RETURN_TYPES = ("FILE_3D",)
    INPUT_NAMES = {
        "format_mesh_style": "format.mesh_style",
        "format_mesh_style_bone_vis": "format.mesh_style.bone_vis",
        "format_mesh_style_bone_vis_bone_vis_radius_m": "format.mesh_style.bone_vis.bone_vis_radius_m",
        "format_mesh_style_bone_vis_bone_vis_color": "format.mesh_style.bone_vis.bone_vis_color",
        "format_mesh_style_shader": "format.mesh_style.shader",
        "format_mesh_style_shader_rainbow_tilt_z": "format.mesh_style.shader.rainbow_tilt_z",
        "format_mesh_style_shader_rainbow_tilt_x": "format.mesh_style.shader.rainbow_tilt_x",
        "format_mesh_style_shader_person_palette_falloff": "format.mesh_style.shader.person_palette_falloff",
        "format_mesh_style_marker_radius_m": "format.mesh_style.marker_radius_m",
        "format_mesh_style_stick_radius_m": "format.mesh_style.stick_radius_m",
        "format_mesh_style_include_hands": "format.mesh_style.include_hands",
        "format_mesh_style_hand_marker_radius_m": "format.mesh_style.hand_marker_radius_m",
        "format_mesh_style_hand_stick_radius_m": "format.mesh_style.hand_stick_radius_m",
        "format_mesh_style_face_style": "format.mesh_style.face_style",
        "format_mesh_style_face_marker_radius_m": "format.mesh_style.face_marker_radius_m",
        "format_mesh_style_material_roughness": "format.mesh_style.material_roughness",
        "format_bone_smooth_window": "format.bone_smooth_window",
        "format_units": "format.units",
    }
    INPUT_SOCKET_TYPES = {
        "pose_data": "MHR_POSE_DATA,KIMODO_POSE_DATA",
        "format": "COMFY_DYNAMICCOMBO_V3",
        "format_mesh_style": "COMFY_DYNAMICCOMBO_V3",
        "format_mesh_style_bone_vis": "COMFY_DYNAMICCOMBO_V3",
        "format_mesh_style_bone_vis_bone_vis_radius_m": "FLOAT",
        "format_mesh_style_bone_vis_bone_vis_color": "COMBO",
        "format_mesh_style_shader": "COMFY_DYNAMICCOMBO_V3",
        "format_mesh_style_shader_rainbow_tilt_z": "FLOAT",
        "format_mesh_style_shader_rainbow_tilt_x": "FLOAT",
        "format_mesh_style_shader_person_palette_falloff": "FLOAT",
        "format_mesh_style_marker_radius_m": "FLOAT",
        "format_mesh_style_stick_radius_m": "FLOAT",
        "format_mesh_style_include_hands": "BOOLEAN",
        "format_mesh_style_hand_marker_radius_m": "FLOAT",
        "format_mesh_style_hand_stick_radius_m": "FLOAT",
        "format_mesh_style_face_style": "COMBO",
        "format_mesh_style_face_marker_radius_m": "FLOAT",
        "format_mesh_style_material_roughness": "FLOAT",
        "format_bone_smooth_window": "INT",
        "format_units": "COMBO",
        "fps": "FLOAT",
        "camera_translation": "COMBO",
        "track_index": "INT",
        "sam3d_body_model": "SAM3D_BODY_MODEL",
    }
    DYNAMIC_INPUTS = {
        "format": Combo(
            {
                "glb": {
                    "format_mesh_style": Combo(
                        {
                            "body_mesh": {
                                "format_mesh_style_bone_vis": Combo(
                                    {
                                        "off": {},
                                        "octahedrons": {
                                            "format_mesh_style_bone_vis_bone_vis_radius_m": Number(
                                                default=0.02, low=0.005, high=0.5
                                            ),
                                            "format_mesh_style_bone_vis_bone_vis_color": Choice(
                                                ("white", "rainbow_y"),
                                                default="rainbow_y",
                                            ),
                                        },
                                    }
                                ),
                                "format_mesh_style_shader": Combo(
                                    {
                                        "default": {},
                                        "rainbow": {
                                            "format_mesh_style_shader_rainbow_tilt_z": Number(
                                                default=-35.0, low=-90.0, high=90.0
                                            ),
                                            "format_mesh_style_shader_rainbow_tilt_x": Number(
                                                default=0.0, low=-90.0, high=90.0
                                            ),
                                            "format_mesh_style_shader_person_palette_falloff": Number(
                                                default=0.6, low=0.1, high=1.0
                                            ),
                                        },
                                        "rainbow_face_normal": {
                                            "format_mesh_style_shader_rainbow_tilt_z": Number(
                                                default=-35.0, low=-90.0, high=90.0
                                            ),
                                            "format_mesh_style_shader_rainbow_tilt_x": Number(
                                                default=0.0, low=-90.0, high=90.0
                                            ),
                                            "format_mesh_style_shader_person_palette_falloff": Number(
                                                default=0.6, low=0.1, high=1.0
                                            ),
                                        },
                                        "rainbow_face_semantic": {
                                            "format_mesh_style_shader_rainbow_tilt_z": Number(
                                                default=-35.0, low=-90.0, high=90.0
                                            ),
                                            "format_mesh_style_shader_rainbow_tilt_x": Number(
                                                default=0.0, low=-90.0, high=90.0
                                            ),
                                            "format_mesh_style_shader_person_palette_falloff": Number(
                                                default=0.6, low=0.1, high=1.0
                                            ),
                                        },
                                    }
                                ),
                            },
                            "bones_only": {
                                "format_mesh_style_bone_vis": Combo(
                                    {
                                        "octahedrons": {
                                            "format_mesh_style_bone_vis_bone_vis_radius_m": Number(
                                                default=0.02, low=0.005, high=0.5
                                            ),
                                            "format_mesh_style_bone_vis_bone_vis_color": Choice(
                                                ("white", "rainbow_y"),
                                                default="rainbow_y",
                                            ),
                                        }
                                    }
                                )
                            },
                            "openpose": {
                                "format_mesh_style_marker_radius_m": Number(
                                    default=0.01, low=0.005, high=0.1
                                ),
                                "format_mesh_style_stick_radius_m": Number(
                                    default=0.008, low=0.002, high=0.05
                                ),
                                "format_mesh_style_include_hands": Flag(default=False),
                                "format_mesh_style_hand_marker_radius_m": Number(
                                    default=0.005, low=0.001, high=0.1
                                ),
                                "format_mesh_style_hand_stick_radius_m": Number(
                                    default=0.003, low=0.001, high=0.05
                                ),
                                "format_mesh_style_face_style": Choice(
                                    ("disabled", "full", "eyes_mouth"),
                                    default="disabled",
                                ),
                                "format_mesh_style_face_marker_radius_m": Number(
                                    default=0.0, low=0.0, high=0.05
                                ),
                            },
                            "scail": {
                                "format_mesh_style_stick_radius_m": Number(
                                    default=0.022, low=0.002, high=0.1
                                ),
                                "format_mesh_style_marker_radius_m": Number(
                                    default=0.0, low=0.0, high=0.1
                                ),
                                "format_mesh_style_material_roughness": Number(
                                    default=0.3, low=0.0, high=1.0
                                ),
                                "format_mesh_style_include_hands": Flag(default=False),
                                "format_mesh_style_hand_marker_radius_m": Number(
                                    default=0.005, low=0.001, high=0.05
                                ),
                                "format_mesh_style_hand_stick_radius_m": Number(
                                    default=0.003, low=0.001, high=0.05
                                ),
                                "format_mesh_style_face_style": Choice(
                                    ("disabled", "full", "eyes_mouth"),
                                    default="disabled",
                                ),
                            },
                        }
                    ),
                    "format_bone_smooth_window": Number(default=0, low=0, high=51),
                },
                "bvh": {"format_units": Choice(("cm", "m"), default="cm")},
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        pose_data: tuple[str, int],
        format: BuildPoseFile_Formats = "glb",
        format_mesh_style: BuildPoseFile_Format_Mesh_Styles | None = None,
        format_mesh_style_bone_vis: (
            BuildPoseFile_Format_Mesh_Style_Bone_Vis | None
        ) = None,
        format_mesh_style_bone_vis_bone_vis_radius_m: (
            float | tuple[str, int] | None
        ) = None,
        format_mesh_style_bone_vis_bone_vis_color: (
            BuildPoseFile_Format_Mesh_Style_Bone_Vis_Bone_Vis_Colors
            | tuple[str, int]
            | None
        ) = None,
        format_mesh_style_shader: BuildPoseFile_Format_Mesh_Style_Shaders | None = None,
        format_mesh_style_shader_rainbow_tilt_z: float | tuple[str, int] | None = None,
        format_mesh_style_shader_rainbow_tilt_x: float | tuple[str, int] | None = None,
        format_mesh_style_shader_person_palette_falloff: (
            float | tuple[str, int] | None
        ) = None,
        format_mesh_style_marker_radius_m: float | tuple[str, int] | None = None,
        format_mesh_style_stick_radius_m: float | tuple[str, int] | None = None,
        format_mesh_style_include_hands: bool | tuple[str, int] | None = None,
        format_mesh_style_hand_marker_radius_m: float | tuple[str, int] | None = None,
        format_mesh_style_hand_stick_radius_m: float | tuple[str, int] | None = None,
        format_mesh_style_face_style: (
            BuildPoseFile_Format_Mesh_Style_Face_Styles | tuple[str, int] | None
        ) = None,
        format_mesh_style_face_marker_radius_m: float | tuple[str, int] | None = None,
        format_mesh_style_material_roughness: float | tuple[str, int] | None = None,
        format_bone_smooth_window: int | tuple[str, int] | None = None,
        format_units: BuildPoseFile_Format_Units | tuple[str, int] | None = None,
        fps: float | tuple[str, int] = 24.0,
        camera_translation: BuildPoseFile_Camera_Translations | tuple[str, int] = "off",
        track_index: int | tuple[str, int] = -1,
        sam3d_body_model: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.pose_data = pose_data
        self.format = format
        if format_mesh_style is not None:
            self.format_mesh_style = format_mesh_style
        if format_mesh_style_bone_vis is not None:
            self.format_mesh_style_bone_vis = format_mesh_style_bone_vis
        if format_mesh_style_bone_vis_bone_vis_radius_m is not None:
            self.format_mesh_style_bone_vis_bone_vis_radius_m = (
                format_mesh_style_bone_vis_bone_vis_radius_m
            )
        if format_mesh_style_bone_vis_bone_vis_color is not None:
            self.format_mesh_style_bone_vis_bone_vis_color = (
                format_mesh_style_bone_vis_bone_vis_color
            )
        if format_mesh_style_shader is not None:
            self.format_mesh_style_shader = format_mesh_style_shader
        if format_mesh_style_shader_rainbow_tilt_z is not None:
            self.format_mesh_style_shader_rainbow_tilt_z = (
                format_mesh_style_shader_rainbow_tilt_z
            )
        if format_mesh_style_shader_rainbow_tilt_x is not None:
            self.format_mesh_style_shader_rainbow_tilt_x = (
                format_mesh_style_shader_rainbow_tilt_x
            )
        if format_mesh_style_shader_person_palette_falloff is not None:
            self.format_mesh_style_shader_person_palette_falloff = (
                format_mesh_style_shader_person_palette_falloff
            )
        if format_mesh_style_marker_radius_m is not None:
            self.format_mesh_style_marker_radius_m = format_mesh_style_marker_radius_m
        if format_mesh_style_stick_radius_m is not None:
            self.format_mesh_style_stick_radius_m = format_mesh_style_stick_radius_m
        if format_mesh_style_include_hands is not None:
            self.format_mesh_style_include_hands = format_mesh_style_include_hands
        if format_mesh_style_hand_marker_radius_m is not None:
            self.format_mesh_style_hand_marker_radius_m = (
                format_mesh_style_hand_marker_radius_m
            )
        if format_mesh_style_hand_stick_radius_m is not None:
            self.format_mesh_style_hand_stick_radius_m = (
                format_mesh_style_hand_stick_radius_m
            )
        if format_mesh_style_face_style is not None:
            self.format_mesh_style_face_style = format_mesh_style_face_style
        if format_mesh_style_face_marker_radius_m is not None:
            self.format_mesh_style_face_marker_radius_m = (
                format_mesh_style_face_marker_radius_m
            )
        if format_mesh_style_material_roughness is not None:
            self.format_mesh_style_material_roughness = (
                format_mesh_style_material_roughness
            )
        if format_bone_smooth_window is not None:
            self.format_bone_smooth_window = format_bone_smooth_window
        if format_units is not None:
            self.format_units = format_units
        self.fps = fps
        self.camera_translation = camera_translation
        self.track_index = track_index
        if sam3d_body_model is not None:
            self.sam3d_body_model = sam3d_body_model
        self.__validate()

    def __validate(self) -> None:
        validate_range("fps", self.fps, 1.0, 240.0)
        validate_choice(
            "camera_translation",
            self.camera_translation,
            BUILDPOSEFILE_CAMERA_TRANSLATIONS,
        )
        validate_range("track_index", self.track_index, -1, 15)
        validate_dynamic(self)


CreateCameraInfo_Modes = Literal["orbit", "look_at", "quaternion"]


CreateCameraInfo_Camera_Types = Literal["perspective", "orthographic"]


CREATECAMERAINFO_CAMERA_TYPES = frozenset(get_args(CreateCameraInfo_Camera_Types))


class CreateCameraInfo(DynamicInputs, BaseNode):
    __slots__ = (
        "mode",
        "mode_yaw",
        "mode_pitch",
        "mode_distance",
        "mode_position_x",
        "mode_position_y",
        "mode_position_z",
        "mode_quat_x",
        "mode_quat_y",
        "mode_quat_z",
        "mode_quat_w",
        "target_x",
        "target_y",
        "target_z",
        "roll",
        "fov",
        "zoom",
        "camera_type",
        "camera_info_state",
    )
    CATEGORY = "3d"
    OUTPUTS = {"camera_info": 0}
    RETURN_TYPES = ("LOAD3D_CAMERA",)
    INPUT_NAMES = {
        "mode_yaw": "mode.yaw",
        "mode_pitch": "mode.pitch",
        "mode_distance": "mode.distance",
        "mode_position_x": "mode.position_x",
        "mode_position_y": "mode.position_y",
        "mode_position_z": "mode.position_z",
        "mode_quat_x": "mode.quat_x",
        "mode_quat_y": "mode.quat_y",
        "mode_quat_z": "mode.quat_z",
        "mode_quat_w": "mode.quat_w",
    }
    INPUT_SOCKET_TYPES = {
        "mode": "COMFY_DYNAMICCOMBO_V3",
        "mode_yaw": "FLOAT",
        "mode_pitch": "FLOAT",
        "mode_distance": "FLOAT",
        "mode_position_x": "FLOAT",
        "mode_position_y": "FLOAT",
        "mode_position_z": "FLOAT",
        "mode_quat_x": "FLOAT",
        "mode_quat_y": "FLOAT",
        "mode_quat_z": "FLOAT",
        "mode_quat_w": "FLOAT",
        "target_x": "FLOAT",
        "target_y": "FLOAT",
        "target_z": "FLOAT",
        "roll": "FLOAT",
        "fov": "FLOAT",
        "zoom": "FLOAT",
        "camera_type": "COMBO",
        "camera_info_state": "CAMERA_INFO_STATE",
    }
    DYNAMIC_INPUTS = {
        "mode": Combo(
            {
                "orbit": {
                    "mode_yaw": Number(default=35.0, low=-360.0, high=360.0),
                    "mode_pitch": Number(default=30.0, low=-89.0, high=89.0),
                    "mode_distance": Number(default=4.0, low=0.01, high=1000.0),
                },
                "look_at": {
                    "mode_position_x": Number(default=4.0, low=-1000.0, high=1000.0),
                    "mode_position_y": Number(default=4.0, low=-1000.0, high=1000.0),
                    "mode_position_z": Number(default=4.0, low=-1000.0, high=1000.0),
                },
                "quaternion": {
                    "mode_position_x": Number(default=4.0, low=-1000.0, high=1000.0),
                    "mode_position_y": Number(default=4.0, low=-1000.0, high=1000.0),
                    "mode_position_z": Number(default=4.0, low=-1000.0, high=1000.0),
                    "mode_quat_x": Number(default=0.0, low=-1.0, high=1.0),
                    "mode_quat_y": Number(default=0.0, low=-1.0, high=1.0),
                    "mode_quat_z": Number(default=0.0, low=-1.0, high=1.0),
                    "mode_quat_w": Number(default=1.0, low=-1.0, high=1.0),
                },
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        camera_type: CreateCameraInfo_Camera_Types | tuple[str, int],
        mode: CreateCameraInfo_Modes = "orbit",
        mode_yaw: float | tuple[str, int] | None = None,
        mode_pitch: float | tuple[str, int] | None = None,
        mode_distance: float | tuple[str, int] | None = None,
        mode_position_x: float | tuple[str, int] | None = None,
        mode_position_y: float | tuple[str, int] | None = None,
        mode_position_z: float | tuple[str, int] | None = None,
        mode_quat_x: float | tuple[str, int] | None = None,
        mode_quat_y: float | tuple[str, int] | None = None,
        mode_quat_z: float | tuple[str, int] | None = None,
        mode_quat_w: float | tuple[str, int] | None = None,
        target_x: float | tuple[str, int] = 0.0,
        target_y: float | tuple[str, int] = 0.0,
        target_z: float | tuple[str, int] = 0.0,
        roll: float | tuple[str, int] = 0.0,
        fov: float | tuple[str, int] = 35.0,
        zoom: float | tuple[str, int] = 1.0,
        camera_info_state: Any = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mode = mode
        if mode_yaw is not None:
            self.mode_yaw = mode_yaw
        if mode_pitch is not None:
            self.mode_pitch = mode_pitch
        if mode_distance is not None:
            self.mode_distance = mode_distance
        if mode_position_x is not None:
            self.mode_position_x = mode_position_x
        if mode_position_y is not None:
            self.mode_position_y = mode_position_y
        if mode_position_z is not None:
            self.mode_position_z = mode_position_z
        if mode_quat_x is not None:
            self.mode_quat_x = mode_quat_x
        if mode_quat_y is not None:
            self.mode_quat_y = mode_quat_y
        if mode_quat_z is not None:
            self.mode_quat_z = mode_quat_z
        if mode_quat_w is not None:
            self.mode_quat_w = mode_quat_w
        self.target_x = target_x
        self.target_y = target_y
        self.target_z = target_z
        self.roll = roll
        self.fov = fov
        self.zoom = zoom
        self.camera_type = camera_type
        if camera_info_state is not None:
            self.camera_info_state = camera_info_state
        self.__validate()

    def __validate(self) -> None:
        validate_range("target_x", self.target_x, -1000.0, 1000.0)
        validate_range("target_y", self.target_y, -1000.0, 1000.0)
        validate_range("target_z", self.target_z, -1000.0, 1000.0)
        validate_range("roll", self.roll, -180.0, 180.0)
        validate_range("fov", self.fov, 1.0, 120.0)
        validate_range("zoom", self.zoom, 0.01, 100.0)
        validate_choice("camera_type", self.camera_type, CREATECAMERAINFO_CAMERA_TYPES)
        validate_dynamic(self)


__getattr__ = submodules_of(__name__)
