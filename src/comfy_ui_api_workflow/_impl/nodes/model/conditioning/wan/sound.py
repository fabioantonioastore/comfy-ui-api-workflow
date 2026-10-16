from ....base import BaseNode
from ....validation import validate_range


class WanSoundImageToVideo(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "audio_encoder_output",
        "ref_image",
        "control_video",
        "ref_motion",
    )
    CATEGORY = "model/conditioning/wan/sound"
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
        "audio_encoder_output": "AUDIO_ENCODER_OUTPUT",
        "ref_image": "IMAGE",
        "control_video": "IMAGE",
        "ref_motion": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        vae: tuple[str, int],
        width: int | tuple[str, int] = 832,
        height: int | tuple[str, int] = 480,
        length: int | tuple[str, int] = 77,
        batch_size: int | tuple[str, int] = 1,
        audio_encoder_output: tuple[str, int] | None = None,
        ref_image: tuple[str, int] | None = None,
        control_video: tuple[str, int] | None = None,
        ref_motion: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        if audio_encoder_output is not None:
            self.audio_encoder_output = audio_encoder_output
        if ref_image is not None:
            self.ref_image = ref_image
        if control_video is not None:
            self.control_video = control_video
        if ref_motion is not None:
            self.ref_motion = ref_motion
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)


class WanSoundImageToVideoExtend(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "video_latent",
        "length",
        "audio_encoder_output",
        "ref_image",
        "control_video",
    )
    CATEGORY = "model/conditioning/wan/sound"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "length": "INT",
        "video_latent": "LATENT",
        "audio_encoder_output": "AUDIO_ENCODER_OUTPUT",
        "ref_image": "IMAGE",
        "control_video": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        vae: tuple[str, int],
        video_latent: tuple[str, int],
        length: int | tuple[str, int] = 77,
        audio_encoder_output: tuple[str, int] | None = None,
        ref_image: tuple[str, int] | None = None,
        control_video: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.video_latent = video_latent
        self.length = length
        if audio_encoder_output is not None:
            self.audio_encoder_output = audio_encoder_output
        if ref_image is not None:
            self.ref_image = ref_image
        if control_video is not None:
            self.control_video = control_video
        self.__validate()

    def __validate(self) -> None:
        validate_range("length", self.length, 1, 16_384)
