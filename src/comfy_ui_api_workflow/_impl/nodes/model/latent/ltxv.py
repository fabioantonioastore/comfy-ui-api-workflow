from ...base import BaseNode
from ...validation import validate_range


class LTXVLatentUpsampler(BaseNode):
    __slots__ = ("samples", "upscale_model", "vae")
    CATEGORY = "model/latent/ltxv"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "samples": "LATENT",
        "upscale_model": "LATENT_UPSCALE_MODEL",
        "vae": "VAE",
    }

    def __init__(
        self,
        node_id: str,
        samples: tuple[str, int],
        upscale_model: tuple[str, int],
        vae: tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.upscale_model = upscale_model
        self.vae = vae


class LTXVAudioVAEEncode(BaseNode):
    __slots__ = ("audio", "audio_vae")
    CATEGORY = "model/latent/ltxv"
    OUTPUTS = {"Audio Latent": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"audio": "AUDIO", "audio_vae": "VAE"}

    def __init__(
        self, node_id: str, audio: tuple[str, int], audio_vae: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.audio = audio
        self.audio_vae = audio_vae


class LTXVAudioVAEDecode(BaseNode):
    __slots__ = ("samples", "audio_vae")
    CATEGORY = "model/latent/ltxv"
    OUTPUTS = {"Audio": 0}
    RETURN_TYPES = ("AUDIO",)
    INPUT_SOCKET_TYPES = {"samples": "LATENT", "audio_vae": "VAE"}

    def __init__(
        self, node_id: str, samples: tuple[str, int], audio_vae: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.audio_vae = audio_vae


class LTXVEmptyLatentAudio(BaseNode):
    __slots__ = ("audio_vae", "frames_number", "frame_rate", "batch_size")
    CATEGORY = "model/latent/ltxv"
    OUTPUTS = {"Latent": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "frames_number": "INT",
        "frame_rate": "FLOAT,INT",
        "batch_size": "INT",
        "audio_vae": "VAE",
    }

    def __init__(
        self,
        node_id: str,
        audio_vae: tuple[str, int],
        frames_number: int | tuple[str, int] = 97,
        frame_rate: float | tuple[str, int] = 25.0,
        batch_size: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.audio_vae = audio_vae
        self.frames_number = frames_number
        self.frame_rate = frame_rate
        self.batch_size = batch_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("frames_number", self.frames_number, 1, 1000)
        validate_range("frame_rate", self.frame_rate, 1.0, 1000.0)
        validate_range("batch_size", self.batch_size, 1, 4096)


class EmptyLTXVLatentVideo(BaseNode):
    __slots__ = ("width", "height", "length", "batch_size")
    CATEGORY = "model/latent/ltxv"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
    }

    def __init__(
        self,
        node_id: str,
        width: int | tuple[str, int] = 768,
        height: int | tuple[str, int] = 512,
        length: int | tuple[str, int] = 97,
        batch_size: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 64, 16_384)
        validate_range("height", self.height, 64, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)


class LTXVConcatAVLatent(BaseNode):
    __slots__ = ("video_latent", "audio_latent")
    CATEGORY = "model/latent/ltxv"
    OUTPUTS = {"latent": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"video_latent": "LATENT", "audio_latent": "LATENT"}

    def __init__(
        self, node_id: str, video_latent: tuple[str, int], audio_latent: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.video_latent = video_latent
        self.audio_latent = audio_latent


class LTXVSeparateAVLatent(BaseNode):
    __slots__ = ("av_latent",)
    CATEGORY = "model/latent/ltxv"
    OUTPUTS = {"video_latent": 0, "audio_latent": 1}
    RETURN_TYPES = ("LATENT", "LATENT")
    INPUT_SOCKET_TYPES = {"av_latent": "LATENT"}

    def __init__(self, node_id: str, av_latent: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.av_latent = av_latent


class LTXVFreezeLatent(BaseNode):
    __slots__ = ("latent",)
    CATEGORY = "model/latent/ltxv"
    OUTPUTS = {"latent": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"latent": "LATENT"}

    def __init__(self, node_id: str, latent: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.latent = latent
