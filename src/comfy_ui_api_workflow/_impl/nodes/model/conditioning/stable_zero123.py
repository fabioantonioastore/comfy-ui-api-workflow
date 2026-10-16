from ...base import BaseNode
from ...validation import validate_range


class StableZero123_Conditioning(BaseNode):
    __slots__ = (
        "clip_vision",
        "init_image",
        "vae",
        "width",
        "height",
        "batch_size",
        "elevation",
        "azimuth",
    )
    CATEGORY = "model/conditioning/stable zero123"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "clip_vision": "CLIP_VISION",
        "init_image": "IMAGE",
        "vae": "VAE",
        "width": "INT",
        "height": "INT",
        "batch_size": "INT",
        "elevation": "FLOAT",
        "azimuth": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        clip_vision: tuple[str, int],
        init_image: tuple[str, int],
        vae: tuple[str, int],
        width: int | tuple[str, int] = 256,
        height: int | tuple[str, int] = 256,
        batch_size: int | tuple[str, int] = 1,
        elevation: float | tuple[str, int] = 0.0,
        azimuth: float | tuple[str, int] = 0.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip_vision = clip_vision
        self.init_image = init_image
        self.vae = vae
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.elevation = elevation
        self.azimuth = azimuth
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)
        validate_range("elevation", self.elevation, -180.0, 180.0)
        validate_range("azimuth", self.azimuth, -180.0, 180.0)


class StableZero123_Conditioning_Batched(BaseNode):
    __slots__ = (
        "clip_vision",
        "init_image",
        "vae",
        "width",
        "height",
        "batch_size",
        "elevation",
        "azimuth",
        "elevation_batch_increment",
        "azimuth_batch_increment",
    )
    CATEGORY = "model/conditioning/stable zero123"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "clip_vision": "CLIP_VISION",
        "init_image": "IMAGE",
        "vae": "VAE",
        "width": "INT",
        "height": "INT",
        "batch_size": "INT",
        "elevation": "FLOAT",
        "azimuth": "FLOAT",
        "elevation_batch_increment": "FLOAT",
        "azimuth_batch_increment": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        clip_vision: tuple[str, int],
        init_image: tuple[str, int],
        vae: tuple[str, int],
        width: int | tuple[str, int] = 256,
        height: int | tuple[str, int] = 256,
        batch_size: int | tuple[str, int] = 1,
        elevation: float | tuple[str, int] = 0.0,
        azimuth: float | tuple[str, int] = 0.0,
        elevation_batch_increment: float | tuple[str, int] = 0.0,
        azimuth_batch_increment: float | tuple[str, int] = 0.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip_vision = clip_vision
        self.init_image = init_image
        self.vae = vae
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.elevation = elevation
        self.azimuth = azimuth
        self.elevation_batch_increment = elevation_batch_increment
        self.azimuth_batch_increment = azimuth_batch_increment
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)
        validate_range("elevation", self.elevation, -180.0, 180.0)
        validate_range("azimuth", self.azimuth, -180.0, 180.0)
        validate_range(
            "elevation_batch_increment", self.elevation_batch_increment, -180.0, 180.0
        )
        validate_range(
            "azimuth_batch_increment", self.azimuth_batch_increment, -180.0, 180.0
        )
