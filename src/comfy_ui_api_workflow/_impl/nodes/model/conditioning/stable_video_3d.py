from ...base import BaseNode
from ...validation import validate_range


class SV3D_Conditioning(BaseNode):
    __slots__ = (
        "clip_vision",
        "init_image",
        "vae",
        "width",
        "height",
        "video_frames",
        "elevation",
    )
    CATEGORY = "model/conditioning/stable video 3d"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "clip_vision": "CLIP_VISION",
        "init_image": "IMAGE",
        "vae": "VAE",
        "width": "INT",
        "height": "INT",
        "video_frames": "INT",
        "elevation": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        clip_vision: tuple[str, int],
        init_image: tuple[str, int],
        vae: tuple[str, int],
        width: int | tuple[str, int] = 576,
        height: int | tuple[str, int] = 576,
        video_frames: int | tuple[str, int] = 21,
        elevation: float | tuple[str, int] = 0.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip_vision = clip_vision
        self.init_image = init_image
        self.vae = vae
        self.width = width
        self.height = height
        self.video_frames = video_frames
        self.elevation = elevation
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("video_frames", self.video_frames, 1, 4096)
        validate_range("elevation", self.elevation, -90.0, 90.0)
