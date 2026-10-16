from ....base import BaseNode
from ....validation import validate_range
from .....lazy import submodules_of


class WanImageToVideo(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "clip_vision_output",
        "start_image",
        "ref_pad_image",
    )
    CATEGORY = "model/conditioning/wan"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
        "clip_vision_output": "CLIP_VISION_OUTPUT",
        "start_image": "IMAGE",
        "ref_pad_image": "IMAGE",
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
        clip_vision_output: tuple[str, int] | None = None,
        start_image: tuple[str, int] | None = None,
        ref_pad_image: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        if clip_vision_output is not None:
            self.clip_vision_output = clip_vision_output
        if start_image is not None:
            self.start_image = start_image
        if ref_pad_image is not None:
            self.ref_pad_image = ref_pad_image
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)


class WanFirstLastFrameToVideo(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "clip_vision_start_image",
        "clip_vision_end_image",
        "start_image",
        "end_image",
    )
    CATEGORY = "model/conditioning/wan"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
        "clip_vision_start_image": "CLIP_VISION_OUTPUT",
        "clip_vision_end_image": "CLIP_VISION_OUTPUT",
        "start_image": "IMAGE",
        "end_image": "IMAGE",
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
        clip_vision_start_image: tuple[str, int] | None = None,
        clip_vision_end_image: tuple[str, int] | None = None,
        start_image: tuple[str, int] | None = None,
        end_image: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        if clip_vision_start_image is not None:
            self.clip_vision_start_image = clip_vision_start_image
        if clip_vision_end_image is not None:
            self.clip_vision_end_image = clip_vision_end_image
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


class Wan22ImageToVideoLatent(BaseNode):
    __slots__ = ("vae", "width", "height", "length", "batch_size", "start_image")
    CATEGORY = "model/conditioning/wan"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
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
        vae: tuple[str, int],
        width: int | tuple[str, int] = 1280,
        height: int | tuple[str, int] = 704,
        length: int | tuple[str, int] = 49,
        batch_size: int | tuple[str, int] = 1,
        start_image: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.vae = vae
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        if start_image is not None:
            self.start_image = start_image
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 32, 16_384)
        validate_range("height", self.height, 32, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)


__getattr__ = submodules_of(__name__)
