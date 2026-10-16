from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice, validate_range


class EmptyHunyuanLatentVideo(BaseNode):
    __slots__ = ("width", "height", "length", "batch_size")
    CATEGORY = "model/latent/hunyuan video"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
    }

    def __init__(
        self,
        node_id: str,
        width: int | tuple[str, int] = 848,
        height: int | tuple[str, int] = 480,
        length: int | tuple[str, int] = 25,
        batch_size: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)


class EmptyHunyuanVideo15Latent(BaseNode):
    __slots__ = ("width", "height", "length", "batch_size")
    CATEGORY = "model/latent/hunyuan video"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
    }

    def __init__(
        self,
        node_id: str,
        width: int | tuple[str, int] = 848,
        height: int | tuple[str, int] = 480,
        length: int | tuple[str, int] = 25,
        batch_size: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)


HunyuanVideo15LatentUpscaleWithModel_Crops = Literal["disabled", "center"]
HunyuanVideo15LatentUpscaleWithModel_Upscale_Methods = Literal[
    "nearest-exact", "bilinear", "area", "bicubic", "bislerp"
]
HUNYUANVIDEO15LATENTUPSCALEWITHMODEL_CROPS = frozenset(
    get_args(HunyuanVideo15LatentUpscaleWithModel_Crops)
)
HUNYUANVIDEO15LATENTUPSCALEWITHMODEL_UPSCALE_METHODS = frozenset(
    get_args(HunyuanVideo15LatentUpscaleWithModel_Upscale_Methods)
)


class HunyuanVideo15LatentUpscaleWithModel(BaseNode):
    __slots__ = ("model", "samples", "crop", "upscale_method", "width", "height")
    CATEGORY = "model/latent/hunyuan video"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "model": "LATENT_UPSCALE_MODEL",
        "samples": "LATENT",
        "upscale_method": "COMBO",
        "width": "INT",
        "height": "INT",
        "crop": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        samples: tuple[str, int],
        crop: HunyuanVideo15LatentUpscaleWithModel_Crops | tuple[str, int],
        upscale_method: (
            HunyuanVideo15LatentUpscaleWithModel_Upscale_Methods | tuple[str, int]
        ) = "bilinear",
        width: int | tuple[str, int] = 1280,
        height: int | tuple[str, int] = 720,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.samples = samples
        self.crop = crop
        self.upscale_method = upscale_method
        self.width = width
        self.height = height
        self.__validate()

    def __validate(self) -> None:
        validate_choice("crop", self.crop, HUNYUANVIDEO15LATENTUPSCALEWITHMODEL_CROPS)
        validate_choice(
            "upscale_method",
            self.upscale_method,
            HUNYUANVIDEO15LATENTUPSCALEWITHMODEL_UPSCALE_METHODS,
        )
        validate_range("width", self.width, 0, 16_384)
        validate_range("height", self.height, 0, 16_384)
