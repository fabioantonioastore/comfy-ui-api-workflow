from typing import Any, Literal, get_args

from ..base import BaseNode
from ..validation import validate_choice, validate_not_empty, validate_range
from ...lazy import submodules_of


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


__getattr__ = submodules_of(__name__)
