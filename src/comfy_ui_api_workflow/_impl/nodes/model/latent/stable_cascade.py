from ...base import BaseNode
from ...validation import validate_range


class StableCascade_EmptyLatentImage(BaseNode):
    __slots__ = ("width", "height", "compression", "batch_size")
    CATEGORY = "model/latent/stable cascade"
    OUTPUTS = {"stage_c": 0, "stage_b": 1}
    RETURN_TYPES = ("LATENT", "LATENT")
    INPUT_SOCKET_TYPES = {
        "width": "INT",
        "height": "INT",
        "compression": "INT",
        "batch_size": "INT",
    }

    def __init__(
        self,
        node_id: str,
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
        compression: int | tuple[str, int] = 42,
        batch_size: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.width = width
        self.height = height
        self.compression = compression
        self.batch_size = batch_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 256, 16_384)
        validate_range("height", self.height, 256, 16_384)
        validate_range("compression", self.compression, 4, 128)
        validate_range("batch_size", self.batch_size, 1, 4096)


class StableCascade_StageC_VAEEncode(BaseNode):
    __slots__ = ("image", "vae", "compression")
    CATEGORY = "model/latent/stable cascade"
    OUTPUTS = {"stage_c": 0, "stage_b": 1}
    RETURN_TYPES = ("LATENT", "LATENT")
    INPUT_SOCKET_TYPES = {"image": "IMAGE", "vae": "VAE", "compression": "INT"}

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        vae: tuple[str, int],
        compression: int | tuple[str, int] = 42,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.vae = vae
        self.compression = compression
        self.__validate()

    def __validate(self) -> None:
        validate_range("compression", self.compression, 4, 128)
