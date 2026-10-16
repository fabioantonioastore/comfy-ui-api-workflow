from typing import Literal, get_args

from ....base import BaseNode
from ....validation import validate_choice, validate_range


class WanAnimateToVideo(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "continue_motion_max_frames",
        "video_frame_offset",
        "clip_vision_output",
        "reference_image",
        "face_video",
        "pose_video",
        "background_video",
        "character_mask",
        "continue_motion",
    )
    CATEGORY = "model/conditioning/wan/animate"
    OUTPUTS = {
        "positive": 0,
        "negative": 1,
        "latent": 2,
        "trim_latent": 3,
        "trim_image": 4,
        "video_frame_offset": 5,
    }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT", "INT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
        "continue_motion_max_frames": "INT",
        "video_frame_offset": "INT",
        "clip_vision_output": "CLIP_VISION_OUTPUT",
        "reference_image": "IMAGE",
        "face_video": "IMAGE",
        "pose_video": "IMAGE",
        "background_video": "IMAGE",
        "character_mask": "MASK",
        "continue_motion": "IMAGE",
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
        continue_motion_max_frames: int | tuple[str, int] = 5,
        video_frame_offset: int | tuple[str, int] = 0,
        clip_vision_output: tuple[str, int] | None = None,
        reference_image: tuple[str, int] | None = None,
        face_video: tuple[str, int] | None = None,
        pose_video: tuple[str, int] | None = None,
        background_video: tuple[str, int] | None = None,
        character_mask: tuple[str, int] | None = None,
        continue_motion: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        self.continue_motion_max_frames = continue_motion_max_frames
        self.video_frame_offset = video_frame_offset
        if clip_vision_output is not None:
            self.clip_vision_output = clip_vision_output
        if reference_image is not None:
            self.reference_image = reference_image
        if face_video is not None:
            self.face_video = face_video
        if pose_video is not None:
            self.pose_video = pose_video
        if background_video is not None:
            self.background_video = background_video
        if character_mask is not None:
            self.character_mask = character_mask
        if continue_motion is not None:
            self.continue_motion = continue_motion
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)
        validate_range(
            "continue_motion_max_frames", self.continue_motion_max_frames, 1, 16_384
        )
        validate_range("video_frame_offset", self.video_frame_offset, 0, 16_384)


class WanAnimate2ToVideo(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "video_frame_offset",
        "pose_strength",
        "pose_start_percent",
        "pose_end_percent",
        "reference_image_strength",
        "reference_image",
        "pose_video",
        "clip_vision_output",
        "positive_pose",
        "clip_vision_output_pose",
        "continue_motion",
    )
    CATEGORY = "model/conditioning/wan/animate"
    OUTPUTS = {
        "positive": 0,
        "negative": 1,
        "latent": 2,
        "trim_latent": 3,
        "trim_image": 4,
        "video_frame_offset": 5,
    }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT", "INT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
        "video_frame_offset": "INT",
        "pose_strength": "FLOAT",
        "pose_start_percent": "FLOAT",
        "pose_end_percent": "FLOAT",
        "reference_image_strength": "FLOAT",
        "reference_image": "IMAGE",
        "pose_video": "IMAGE",
        "clip_vision_output": "CLIP_VISION_OUTPUT",
        "positive_pose": "CONDITIONING",
        "clip_vision_output_pose": "CLIP_VISION_OUTPUT",
        "continue_motion": "IMAGE",
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
        video_frame_offset: int | tuple[str, int] = 0,
        pose_strength: float | tuple[str, int] = 1.0,
        pose_start_percent: float | tuple[str, int] = 0.0,
        pose_end_percent: float | tuple[str, int] = 1.0,
        reference_image_strength: float | tuple[str, int] = 1.0,
        reference_image: tuple[str, int] | None = None,
        pose_video: tuple[str, int] | None = None,
        clip_vision_output: tuple[str, int] | None = None,
        positive_pose: tuple[str, int] | None = None,
        clip_vision_output_pose: tuple[str, int] | None = None,
        continue_motion: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        self.video_frame_offset = video_frame_offset
        self.pose_strength = pose_strength
        self.pose_start_percent = pose_start_percent
        self.pose_end_percent = pose_end_percent
        self.reference_image_strength = reference_image_strength
        if reference_image is not None:
            self.reference_image = reference_image
        if pose_video is not None:
            self.pose_video = pose_video
        if clip_vision_output is not None:
            self.clip_vision_output = clip_vision_output
        if positive_pose is not None:
            self.positive_pose = positive_pose
        if clip_vision_output_pose is not None:
            self.clip_vision_output_pose = clip_vision_output_pose
        if continue_motion is not None:
            self.continue_motion = continue_motion
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)
        validate_range("video_frame_offset", self.video_frame_offset, 0, 16_384)
        validate_range("pose_strength", self.pose_strength, 0.0, 10.0)
        validate_range("pose_start_percent", self.pose_start_percent, 0.0, 1.0)
        validate_range("pose_end_percent", self.pose_end_percent, 0.0, 1.0)
        validate_range(
            "reference_image_strength", self.reference_image_strength, 0.0, 10.0
        )


WanAnimate2Cache_Devices = Literal["cpu", "gpu"]
WanAnimate2Cache_Dtypes = Literal["default", "int8", "int4"]
WANANIMATE2CACHE_DEVICES = frozenset(get_args(WanAnimate2Cache_Devices))
WANANIMATE2CACHE_DTYPES = frozenset(get_args(WanAnimate2Cache_Dtypes))


class WanAnimate2Cache(BaseNode):
    __slots__ = ("model", "device", "dtype")
    CATEGORY = "model/conditioning/wan/animate"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "device": "COMBO", "dtype": "COMBO"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        device: WanAnimate2Cache_Devices | tuple[str, int] = "cpu",
        dtype: WanAnimate2Cache_Dtypes | tuple[str, int] = "default",
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.device = device
        self.dtype = dtype
        self.__validate()

    def __validate(self) -> None:
        validate_choice("device", self.device, WANANIMATE2CACHE_DEVICES)
        validate_choice("dtype", self.dtype, WANANIMATE2CACHE_DTYPES)
