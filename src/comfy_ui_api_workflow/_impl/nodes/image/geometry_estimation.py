from typing import Literal, get_args

from ..base import BaseNode
from ..validation import validate_bool, validate_choice, validate_range


class MoGeInference(BaseNode):
    __slots__ = (
        "moge_model",
        "image",
        "resolution_level",
        "fov_x_degrees",
        "batch_size",
        "force_projection",
        "apply_mask",
        "refine_steps",
    )
    CATEGORY = "image/geometry estimation"
    OUTPUTS = {"moge_geometry": 0}
    RETURN_TYPES = ("MOGE_GEOMETRY",)
    INPUT_SOCKET_TYPES = {
        "moge_model": "MOGE_MODEL",
        "image": "IMAGE",
        "resolution_level": "INT",
        "fov_x_degrees": "FLOAT",
        "batch_size": "INT",
        "force_projection": "BOOLEAN",
        "apply_mask": "BOOLEAN",
        "refine_steps": "INT",
    }

    def __init__(
        self,
        node_id: str,
        moge_model: tuple[str, int],
        image: tuple[str, int],
        resolution_level: int | tuple[str, int] = 9,
        fov_x_degrees: float | tuple[str, int] = 0.0,
        batch_size: int | tuple[str, int] = 4,
        force_projection: bool | tuple[str, int] = True,
        apply_mask: bool | tuple[str, int] = True,
        refine_steps: int | tuple[str, int] = 3,
    ) -> None:
        super().__init__(node_id=node_id)
        self.moge_model = moge_model
        self.image = image
        self.resolution_level = resolution_level
        self.fov_x_degrees = fov_x_degrees
        self.batch_size = batch_size
        self.force_projection = force_projection
        self.apply_mask = apply_mask
        self.refine_steps = refine_steps
        self.__validate()

    def __validate(self) -> None:
        validate_range("resolution_level", self.resolution_level, 0, 9)
        validate_range("fov_x_degrees", self.fov_x_degrees, 0.0, 170.0)
        validate_range("batch_size", self.batch_size, 1, 64)
        validate_bool("force_projection", self.force_projection)
        validate_bool("apply_mask", self.apply_mask)
        validate_range("refine_steps", self.refine_steps, 0, 8)


class MoGePanoramaInference(BaseNode):
    __slots__ = (
        "moge_model",
        "image",
        "resolution_level",
        "split_resolution",
        "merge_resolution",
        "batch_size",
        "refine_steps",
    )
    CATEGORY = "image/geometry estimation"
    OUTPUTS = {"moge_geometry": 0}
    RETURN_TYPES = ("MOGE_GEOMETRY",)
    INPUT_SOCKET_TYPES = {
        "moge_model": "MOGE_MODEL",
        "image": "IMAGE",
        "resolution_level": "INT",
        "split_resolution": "INT",
        "merge_resolution": "INT",
        "batch_size": "INT",
        "refine_steps": "INT",
    }

    def __init__(
        self,
        node_id: str,
        moge_model: tuple[str, int],
        image: tuple[str, int],
        resolution_level: int | tuple[str, int] = 9,
        split_resolution: int | tuple[str, int] = 512,
        merge_resolution: int | tuple[str, int] = 1920,
        batch_size: int | tuple[str, int] = 4,
        refine_steps: int | tuple[str, int] = 3,
    ) -> None:
        super().__init__(node_id=node_id)
        self.moge_model = moge_model
        self.image = image
        self.resolution_level = resolution_level
        self.split_resolution = split_resolution
        self.merge_resolution = merge_resolution
        self.batch_size = batch_size
        self.refine_steps = refine_steps
        self.__validate()

    def __validate(self) -> None:
        validate_range("resolution_level", self.resolution_level, 0, 9)
        validate_range("split_resolution", self.split_resolution, 256, 1024)
        validate_range("merge_resolution", self.merge_resolution, 256, 8192)
        validate_range("batch_size", self.batch_size, 1, 12)
        validate_range("refine_steps", self.refine_steps, 0, 8)


MoGeRender_Output_S = Literal[
    "depth", "depth_colored", "normal_opengl", "normal_directx", "mask"
]
MOGERENDER_OUTPUT_S = frozenset(get_args(MoGeRender_Output_S))


class MoGeRender(BaseNode):
    __slots__ = ("moge_geometry", "output_")
    CATEGORY = "image/geometry estimation"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"moge_geometry": "MOGE_GEOMETRY", "output_": "COMBO"}
    INPUT_NAMES = {"output_": "output"}

    def __init__(
        self,
        node_id: str,
        moge_geometry: tuple[str, int],
        output_: MoGeRender_Output_S | tuple[str, int] = "depth",
    ) -> None:
        super().__init__(node_id=node_id)
        self.moge_geometry = moge_geometry
        self.output_ = output_
        self.__validate()

    def __validate(self) -> None:
        validate_choice("output_", self.output_, MOGERENDER_OUTPUT_S)


class MoGePointMapToMesh(BaseNode):
    __slots__ = (
        "moge_geometry",
        "batch_index",
        "decimation",
        "discontinuity_threshold",
        "texture",
    )
    CATEGORY = "image/geometry estimation"
    OUTPUTS = {"MESH": 0}
    RETURN_TYPES = ("MESH",)
    INPUT_SOCKET_TYPES = {
        "moge_geometry": "MOGE_GEOMETRY",
        "batch_index": "INT",
        "decimation": "INT",
        "discontinuity_threshold": "FLOAT",
        "texture": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        moge_geometry: tuple[str, int],
        batch_index: int | tuple[str, int] = 0,
        decimation: int | tuple[str, int] = 1,
        discontinuity_threshold: float | tuple[str, int] = 0.04,
        texture: bool | tuple[str, int] = True,
    ) -> None:
        super().__init__(node_id=node_id)
        self.moge_geometry = moge_geometry
        self.batch_index = batch_index
        self.decimation = decimation
        self.discontinuity_threshold = discontinuity_threshold
        self.texture = texture
        self.__validate()

    def __validate(self) -> None:
        validate_range("batch_index", self.batch_index, 0, 4096)
        validate_range("decimation", self.decimation, 1, 8)
        validate_range(
            "discontinuity_threshold", self.discontinuity_threshold, 0.0, 1.0
        )
        validate_bool("texture", self.texture)


MoGeGeometryToFOV_Axises = Literal["vertical", "horizontal", "diagonal"]
MoGeGeometryToFOV_Units = Literal["degrees", "radians"]
MOGEGEOMETRYTOFOV_AXISES = frozenset(get_args(MoGeGeometryToFOV_Axises))
MOGEGEOMETRYTOFOV_UNITS = frozenset(get_args(MoGeGeometryToFOV_Units))


class MoGeGeometryToFOV(BaseNode):
    __slots__ = ("moge_geometry", "axis", "unit")
    CATEGORY = "image/geometry estimation"
    OUTPUTS = {"fov": 0, "focal_pixels": 1}
    RETURN_TYPES = ("FLOAT", "FLOAT")
    INPUT_SOCKET_TYPES = {
        "moge_geometry": "MOGE_GEOMETRY",
        "axis": "COMBO",
        "unit": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        moge_geometry: tuple[str, int],
        axis: MoGeGeometryToFOV_Axises | tuple[str, int] = "vertical",
        unit: MoGeGeometryToFOV_Units | tuple[str, int] = "degrees",
    ) -> None:
        super().__init__(node_id=node_id)
        self.moge_geometry = moge_geometry
        self.axis = axis
        self.unit = unit
        self.__validate()

    def __validate(self) -> None:
        validate_choice("axis", self.axis, MOGEGEOMETRYTOFOV_AXISES)
        validate_choice("unit", self.unit, MOGEGEOMETRYTOFOV_UNITS)


class DA3GeometryToMesh(BaseNode):
    __slots__ = (
        "da3_geometry",
        "batch_index",
        "decimation",
        "discontinuity_threshold",
        "confidence_threshold",
        "use_sky_mask",
        "texture",
    )
    CATEGORY = "image/geometry estimation"
    OUTPUTS = {"MESH": 0}
    RETURN_TYPES = ("MESH",)
    INPUT_SOCKET_TYPES = {
        "da3_geometry": "DA3_GEOMETRY",
        "batch_index": "INT",
        "decimation": "INT",
        "discontinuity_threshold": "FLOAT",
        "confidence_threshold": "FLOAT",
        "use_sky_mask": "BOOLEAN",
        "texture": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        da3_geometry: tuple[str, int],
        batch_index: int | tuple[str, int] = 0,
        decimation: int | tuple[str, int] = 1,
        discontinuity_threshold: float | tuple[str, int] = 0.04,
        confidence_threshold: float | tuple[str, int] = 0.1,
        use_sky_mask: bool | tuple[str, int] = True,
        texture: bool | tuple[str, int] = True,
    ) -> None:
        super().__init__(node_id=node_id)
        self.da3_geometry = da3_geometry
        self.batch_index = batch_index
        self.decimation = decimation
        self.discontinuity_threshold = discontinuity_threshold
        self.confidence_threshold = confidence_threshold
        self.use_sky_mask = use_sky_mask
        self.texture = texture
        self.__validate()

    def __validate(self) -> None:
        validate_range("batch_index", self.batch_index, 0, 4096)
        validate_range("decimation", self.decimation, 1, 8)
        validate_range(
            "discontinuity_threshold", self.discontinuity_threshold, 0.0, 1.0
        )
        validate_range("confidence_threshold", self.confidence_threshold, 0.0, 1.0)
        validate_bool("use_sky_mask", self.use_sky_mask)
        validate_bool("texture", self.texture)


MarigoldV2PostProcess_Predictions = Literal["depth", "normals", "albedo"]
MARIGOLDV2POSTPROCESS_PREDICTIONS = frozenset(
    get_args(MarigoldV2PostProcess_Predictions)
)


class MarigoldV2PostProcess(BaseNode):
    __slots__ = ("image", "prediction")
    CATEGORY = "image/geometry estimation"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"image": "IMAGE", "prediction": "COMBO"}

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        prediction: MarigoldV2PostProcess_Predictions | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.prediction = prediction
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "prediction", self.prediction, MARIGOLDV2POSTPROCESS_PREDICTIONS
        )
