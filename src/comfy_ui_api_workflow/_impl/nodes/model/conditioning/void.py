from ...base import BaseNode
from ...validation import validate_range


class VOIDInpaintConditioning(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "video",
        "quadmask",
        "width",
        "height",
        "length",
        "batch_size",
    )
    CATEGORY = "model/conditioning/void"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "video": "IMAGE",
        "quadmask": "MASK",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        vae: tuple[str, int],
        video: tuple[str, int],
        quadmask: tuple[str, int],
        width: int | tuple[str, int] = 672,
        height: int | tuple[str, int] = 384,
        length: int | tuple[str, int] = 45,
        batch_size: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.video = video
        self.quadmask = quadmask
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 64)
