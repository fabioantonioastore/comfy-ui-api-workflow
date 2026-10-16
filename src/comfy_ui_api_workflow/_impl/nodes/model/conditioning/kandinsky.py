from ...base import BaseNode
from ...validation import validate_range


class Kandinsky5ImageToVideo(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "start_image",
    )
    CATEGORY = "model/conditioning/kandinsky"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2, "cond_latent": 3}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
        "start_image": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        vae: tuple[str, int],
        width: int | tuple[str, int] = 768,
        height: int | tuple[str, int] = 512,
        length: int | tuple[str, int] = 121,
        batch_size: int | tuple[str, int] = 1,
        start_image: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        if start_image is not None:
            self.start_image = start_image
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)


class CLIPTextEncodeKandinsky5(BaseNode):
    __slots__ = ("clip", "clip_l", "qwen25_7b")
    CATEGORY = "model/conditioning/kandinsky"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {"clip": "CLIP", "clip_l": "STRING", "qwen25_7b": "STRING"}

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        clip_l: str | tuple[str, int],
        qwen25_7b: str | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.clip_l = clip_l
        self.qwen25_7b = qwen25_7b
