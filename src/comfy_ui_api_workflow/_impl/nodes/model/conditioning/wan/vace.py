from ....base import BaseNode
from ....validation import validate_range


class WanVaceToVideo(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "strength",
        "control_video",
        "control_masks",
        "reference_image",
    )
    CATEGORY = "model/conditioning/wan/vace"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2, "trim_latent": 3}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
        "strength": "FLOAT",
        "control_video": "IMAGE",
        "control_masks": "MASK",
        "reference_image": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        vae: tuple[str, int],
        width: int | tuple[str, int] = 832,
        height: int | tuple[str, int] = 480,
        length: int | tuple[str, int] = 81,
        batch_size: int | tuple[str, int] = 1,
        strength: float | tuple[str, int] = 1.0,
        control_video: tuple[str, int] | None = None,
        control_masks: tuple[str, int] | None = None,
        reference_image: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        self.strength = strength
        if control_video is not None:
            self.control_video = control_video
        if control_masks is not None:
            self.control_masks = control_masks
        if reference_image is not None:
            self.reference_image = reference_image
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)
        validate_range("strength", self.strength, 0.0, 1000.0)
