from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice, validate_range


class TextEncodeHunyuanVideo_ImageToVideo(BaseNode):
    __slots__ = ("clip", "clip_vision_output", "prompt", "image_interleave")
    CATEGORY = "model/conditioning/hunyuan video"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "clip_vision_output": "CLIP_VISION_OUTPUT",
        "prompt": "STRING",
        "image_interleave": "INT",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        clip_vision_output: tuple[str, int],
        prompt: str | tuple[str, int],
        image_interleave: int | tuple[str, int] = 2,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.clip_vision_output = clip_vision_output
        self.prompt = prompt
        self.image_interleave = image_interleave
        self.__validate()

    def __validate(self) -> None:
        validate_range("image_interleave", self.image_interleave, 1, 512)


class HunyuanVideo15ImageToVideo(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "start_image",
        "clip_vision_output",
    )
    CATEGORY = "model/conditioning/hunyuan video"
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
        "start_image": "IMAGE",
        "clip_vision_output": "CLIP_VISION_OUTPUT",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        vae: tuple[str, int],
        width: int | tuple[str, int] = 848,
        height: int | tuple[str, int] = 480,
        length: int | tuple[str, int] = 33,
        batch_size: int | tuple[str, int] = 1,
        start_image: tuple[str, int] | None = None,
        clip_vision_output: tuple[str, int] | None = None,
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
        if clip_vision_output is not None:
            self.clip_vision_output = clip_vision_output
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)


class HunyuanVideo15SuperResolution(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "latent",
        "noise_augmentation",
        "vae",
        "start_image",
        "clip_vision_output",
    )
    CATEGORY = "model/conditioning/hunyuan video"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "latent": "LATENT",
        "noise_augmentation": "FLOAT",
        "vae": "VAE",
        "start_image": "IMAGE",
        "clip_vision_output": "CLIP_VISION_OUTPUT",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        latent: tuple[str, int],
        noise_augmentation: float | tuple[str, int] = 0.7,
        vae: tuple[str, int] | None = None,
        start_image: tuple[str, int] | None = None,
        clip_vision_output: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.latent = latent
        self.noise_augmentation = noise_augmentation
        if vae is not None:
            self.vae = vae
        if start_image is not None:
            self.start_image = start_image
        if clip_vision_output is not None:
            self.clip_vision_output = clip_vision_output
        self.__validate()

    def __validate(self) -> None:
        validate_range("noise_augmentation", self.noise_augmentation, 0.0, 1.0)


HunyuanImageToVideo_Guidance_Types = Literal["v1 (concat)", "v2 (replace)", "custom"]
HUNYUANIMAGETOVIDEO_GUIDANCE_TYPES = frozenset(
    get_args(HunyuanImageToVideo_Guidance_Types)
)


class HunyuanImageToVideo(BaseNode):
    __slots__ = (
        "positive",
        "vae",
        "guidance_type",
        "width",
        "height",
        "length",
        "batch_size",
        "start_image",
    )
    CATEGORY = "model/conditioning/hunyuan video"
    OUTPUTS = {"positive": 0, "latent": 1}
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "vae": "VAE",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
        "guidance_type": "COMBO",
        "start_image": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        vae: tuple[str, int],
        guidance_type: HunyuanImageToVideo_Guidance_Types | tuple[str, int],
        width: int | tuple[str, int] = 848,
        height: int | tuple[str, int] = 480,
        length: int | tuple[str, int] = 53,
        batch_size: int | tuple[str, int] = 1,
        start_image: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.vae = vae
        self.guidance_type = guidance_type
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        if start_image is not None:
            self.start_image = start_image
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "guidance_type", self.guidance_type, HUNYUANIMAGETOVIDEO_GUIDANCE_TYPES
        )
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)


class HunyuanRefinerLatent(BaseNode):
    __slots__ = ("positive", "negative", "latent", "noise_augmentation")
    CATEGORY = "model/conditioning/hunyuan video"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "latent": "LATENT",
        "noise_augmentation": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        latent: tuple[str, int],
        noise_augmentation: float | tuple[str, int] = 0.1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.latent = latent
        self.noise_augmentation = noise_augmentation
        self.__validate()

    def __validate(self) -> None:
        validate_range("noise_augmentation", self.noise_augmentation, 0.0, 1.0)
