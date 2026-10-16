from ...base import BaseNode
from ...validation import validate_range


class CosmosImageToVideoLatent(BaseNode):
    __slots__ = (
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "start_image",
        "end_image",
    )
    CATEGORY = "model/conditioning/cosmos"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "vae": "VAE",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
        "start_image": "IMAGE",
        "end_image": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        vae: tuple[str, int],
        width: int | tuple[str, int] = 1280,
        height: int | tuple[str, int] = 704,
        length: int | tuple[str, int] = 121,
        batch_size: int | tuple[str, int] = 1,
        start_image: tuple[str, int] | None = None,
        end_image: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.vae = vae
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        if start_image is not None:
            self.start_image = start_image
        if end_image is not None:
            self.end_image = end_image
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)


class CosmosPredict2ImageToVideoLatent(BaseNode):
    __slots__ = (
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "start_image",
        "end_image",
    )
    CATEGORY = "model/conditioning/cosmos"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "vae": "VAE",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
        "start_image": "IMAGE",
        "end_image": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        vae: tuple[str, int],
        width: int | tuple[str, int] = 848,
        height: int | tuple[str, int] = 480,
        length: int | tuple[str, int] = 93,
        batch_size: int | tuple[str, int] = 1,
        start_image: tuple[str, int] | None = None,
        end_image: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.vae = vae
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        if start_image is not None:
            self.start_image = start_image
        if end_image is not None:
            self.end_image = end_image
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)
