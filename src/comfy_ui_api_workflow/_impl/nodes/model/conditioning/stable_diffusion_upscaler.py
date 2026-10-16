from ...base import BaseNode
from ...validation import validate_range


class SD_4XUpscale_Conditioning(BaseNode):
    __slots__ = ("images", "positive", "negative", "scale_ratio", "noise_augmentation")
    CATEGORY = "model/conditioning/stable diffusion upscaler"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "images": "IMAGE",
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "scale_ratio": "FLOAT",
        "noise_augmentation": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        positive: tuple[str, int],
        negative: tuple[str, int],
        scale_ratio: float | tuple[str, int] = 4.0,
        noise_augmentation: float | tuple[str, int] = 0.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.positive = positive
        self.negative = negative
        self.scale_ratio = scale_ratio
        self.noise_augmentation = noise_augmentation
        self.__validate()

    def __validate(self) -> None:
        validate_range("scale_ratio", self.scale_ratio, 0.0, 10.0)
        validate_range("noise_augmentation", self.noise_augmentation, 0.0, 1.0)
