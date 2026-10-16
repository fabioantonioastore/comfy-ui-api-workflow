from ...base import BaseNode
from ...validation import validate_range


class MiniMaxH3ImageToVideo(BaseNode):
    __slots__ = (
        "clip",
        "vae",
        "prompt",
        "width",
        "height",
        "length",
        "first_frame",
        "last_frame",
    )
    CATEGORY = "model/conditioning/minimax"
    OUTPUTS = {"positive": 0, "LATENT": 1}
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "vae": "VAE",
        "prompt": "STRING",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "first_frame": "IMAGE",
        "last_frame": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        vae: tuple[str, int],
        prompt: str | tuple[str, int],
        width: int | tuple[str, int] = 1344,
        height: int | tuple[str, int] = 768,
        length: int | tuple[str, int] = 124,
        first_frame: tuple[str, int] | None = None,
        last_frame: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.vae = vae
        self.prompt = prompt
        self.width = width
        self.height = height
        self.length = length
        if first_frame is not None:
            self.first_frame = first_frame
        if last_frame is not None:
            self.last_frame = last_frame
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 32, 16_384)
        validate_range("height", self.height, 32, 16_384)
        validate_range("length", self.length, 5, 3600)


class MiniMaxH3AddGuide(BaseNode):
    __slots__ = (
        "positive",
        "latent",
        "frame_idx",
        "vae",
        "audio_vae",
        "image",
        "audio",
    )
    CATEGORY = "model/conditioning/minimax"
    OUTPUTS = {"positive": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "latent": "LATENT",
        "frame_idx": "INT",
        "vae": "VAE",
        "audio_vae": "VAE",
        "image": "IMAGE",
        "audio": "AUDIO",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        latent: tuple[str, int],
        frame_idx: int | tuple[str, int] = 0,
        vae: tuple[str, int] | None = None,
        audio_vae: tuple[str, int] | None = None,
        image: tuple[str, int] | None = None,
        audio: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.latent = latent
        self.frame_idx = frame_idx
        if vae is not None:
            self.vae = vae
        if audio_vae is not None:
            self.audio_vae = audio_vae
        if image is not None:
            self.image = image
        if audio is not None:
            self.audio = audio
        self.__validate()

    def __validate(self) -> None:
        validate_range("frame_idx", self.frame_idx, -9999, 9999)
