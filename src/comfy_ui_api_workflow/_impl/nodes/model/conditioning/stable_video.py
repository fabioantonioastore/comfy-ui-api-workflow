from ...base import BaseNode
from ...validation import validate_range


class SVD_img2vid_Conditioning(BaseNode):
    __slots__ = (
        "clip_vision",
        "init_image",
        "vae",
        "width",
        "height",
        "video_frames",
        "motion_bucket_id",
        "fps",
        "augmentation_level",
    )
    CATEGORY = "model/conditioning/stable video"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "clip_vision": "CLIP_VISION",
        "init_image": "IMAGE",
        "vae": "VAE",
        "width": "INT",
        "height": "INT",
        "video_frames": "INT",
        "motion_bucket_id": "INT",
        "fps": "INT",
        "augmentation_level": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        clip_vision: tuple[str, int],
        init_image: tuple[str, int],
        vae: tuple[str, int],
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 576,
        video_frames: int | tuple[str, int] = 14,
        motion_bucket_id: int | tuple[str, int] = 127,
        fps: int | tuple[str, int] = 6,
        augmentation_level: float | tuple[str, int] = 0.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip_vision = clip_vision
        self.init_image = init_image
        self.vae = vae
        self.width = width
        self.height = height
        self.video_frames = video_frames
        self.motion_bucket_id = motion_bucket_id
        self.fps = fps
        self.augmentation_level = augmentation_level
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("video_frames", self.video_frames, 1, 4096)
        validate_range("motion_bucket_id", self.motion_bucket_id, 1, 1023)
        validate_range("fps", self.fps, 1, 1024)
        validate_range("augmentation_level", self.augmentation_level, 0.0, 10.0)
