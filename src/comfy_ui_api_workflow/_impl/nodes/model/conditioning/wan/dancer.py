from ....base import BaseNode
from ....validation import validate_range


class WanDancerVideo(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "clip_vision_output",
        "clip_vision_output_ref",
        "start_image",
        "mask",
        "audio_encoder_output",
    )
    CATEGORY = "model/conditioning/wan/dancer"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "clip_vision_output": "CLIP_VISION_OUTPUT",
        "clip_vision_output_ref": "CLIP_VISION_OUTPUT",
        "start_image": "IMAGE",
        "mask": "MASK",
        "audio_encoder_output": "AUDIO_ENCODER_OUTPUT",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        vae: tuple[str, int],
        width: int | tuple[str, int] = 480,
        height: int | tuple[str, int] = 832,
        length: int | tuple[str, int] = 149,
        clip_vision_output: tuple[str, int] | None = None,
        clip_vision_output_ref: tuple[str, int] | None = None,
        start_image: tuple[str, int] | None = None,
        mask: tuple[str, int] | None = None,
        audio_encoder_output: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.width = width
        self.height = height
        self.length = length
        if clip_vision_output is not None:
            self.clip_vision_output = clip_vision_output
        if clip_vision_output_ref is not None:
            self.clip_vision_output_ref = clip_vision_output_ref
        if start_image is not None:
            self.start_image = start_image
        if mask is not None:
            self.mask = mask
        if audio_encoder_output is not None:
            self.audio_encoder_output = audio_encoder_output
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("length", self.length, 1, 16_384)


class WanDancerEncodeAudio(BaseNode):
    __slots__ = ("audio", "video_frames", "audio_inject_scale")
    CATEGORY = "model/conditioning/wan/dancer"
    OUTPUTS = {"audio_encoder_output": 0, "fps_string": 1}
    RETURN_TYPES = ("AUDIO_ENCODER_OUTPUT", "STRING")
    INPUT_SOCKET_TYPES = {
        "audio": "AUDIO",
        "video_frames": "INT",
        "audio_inject_scale": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        audio: tuple[str, int],
        video_frames: int | tuple[str, int] = 149,
        audio_inject_scale: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.audio = audio
        self.video_frames = video_frames
        self.audio_inject_scale = audio_inject_scale
        self.__validate()

    def __validate(self) -> None:
        validate_range("video_frames", self.video_frames, 1, 16_384)
        validate_range("audio_inject_scale", self.audio_inject_scale, 0.0, 10.0)
