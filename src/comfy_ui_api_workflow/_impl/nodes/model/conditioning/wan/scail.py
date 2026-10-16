from typing import Literal, get_args

from ....base import BaseNode
from ....validation import validate_bool, validate_choice, validate_range


class WanSCAILToVideo(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "pose_strength",
        "pose_start",
        "pose_end",
        "video_frame_offset",
        "previous_frame_count",
        "pose_video",
        "pose_video_mask",
        "replacement_mode",
        "reference_image",
        "reference_image_mask",
        "clip_vision_output",
        "previous_frames",
    )
    CATEGORY = "model/conditioning/wan/scail"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2, "video_frame_offset": 3}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
        "pose_strength": "FLOAT",
        "pose_start": "FLOAT",
        "pose_end": "FLOAT",
        "video_frame_offset": "INT",
        "previous_frame_count": "INT",
        "pose_video": "IMAGE",
        "pose_video_mask": "IMAGE",
        "replacement_mode": "BOOLEAN",
        "reference_image": "IMAGE",
        "reference_image_mask": "IMAGE",
        "clip_vision_output": "CLIP_VISION_OUTPUT",
        "previous_frames": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        vae: tuple[str, int],
        width: int | tuple[str, int] = 512,
        height: int | tuple[str, int] = 896,
        length: int | tuple[str, int] = 81,
        batch_size: int | tuple[str, int] = 1,
        pose_strength: float | tuple[str, int] = 1.0,
        pose_start: float | tuple[str, int] = 0.0,
        pose_end: float | tuple[str, int] = 1.0,
        video_frame_offset: int | tuple[str, int] = 0,
        previous_frame_count: int | tuple[str, int] = 5,
        pose_video: tuple[str, int] | None = None,
        pose_video_mask: tuple[str, int] | None = None,
        replacement_mode: bool | tuple[str, int] = False,
        reference_image: tuple[str, int] | None = None,
        reference_image_mask: tuple[str, int] | None = None,
        clip_vision_output: tuple[str, int] | None = None,
        previous_frames: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        self.pose_strength = pose_strength
        self.pose_start = pose_start
        self.pose_end = pose_end
        self.video_frame_offset = video_frame_offset
        self.previous_frame_count = previous_frame_count
        if pose_video is not None:
            self.pose_video = pose_video
        if pose_video_mask is not None:
            self.pose_video_mask = pose_video_mask
        self.replacement_mode = replacement_mode
        if reference_image is not None:
            self.reference_image = reference_image
        if reference_image_mask is not None:
            self.reference_image_mask = reference_image_mask
        if clip_vision_output is not None:
            self.clip_vision_output = clip_vision_output
        if previous_frames is not None:
            self.previous_frames = previous_frames
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 32, 16_384)
        validate_range("height", self.height, 32, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)
        validate_range("pose_strength", self.pose_strength, 0.0, 10.0)
        validate_range("pose_start", self.pose_start, 0.0, 1.0)
        validate_range("pose_end", self.pose_end, 0.0, 1.0)
        validate_range("video_frame_offset", self.video_frame_offset, 0, 16_384)
        validate_range("previous_frame_count", self.previous_frame_count, 1, 16_384)
        validate_bool("replacement_mode", self.replacement_mode)


SCAIL2ColoredMask_Sort_Bies = Literal["none", "left_to_right", "area"]
SCAIL2COLOREDMASK_SORT_BIES = frozenset(get_args(SCAIL2ColoredMask_Sort_Bies))


class SCAIL2ColoredMask(BaseNode):
    __slots__ = (
        "driving_track_data",
        "object_indices",
        "sort_by",
        "replacement_mode",
        "ref_track_data",
    )
    CATEGORY = "model/conditioning/wan/scail"
    OUTPUTS = {"pose_video_mask": 0, "reference_image_mask": 1}
    RETURN_TYPES = ("IMAGE", "IMAGE")
    INPUT_SOCKET_TYPES = {
        "driving_track_data": "SAM3_TRACK_DATA",
        "object_indices": "STRING",
        "sort_by": "COMBO",
        "replacement_mode": "BOOLEAN",
        "ref_track_data": "SAM3_TRACK_DATA,MASK",
    }

    def __init__(
        self,
        node_id: str,
        driving_track_data: tuple[str, int],
        object_indices: str | tuple[str, int] = "",
        sort_by: SCAIL2ColoredMask_Sort_Bies | tuple[str, int] = "left_to_right",
        replacement_mode: bool | tuple[str, int] = False,
        ref_track_data: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.driving_track_data = driving_track_data
        self.object_indices = object_indices
        self.sort_by = sort_by
        self.replacement_mode = replacement_mode
        if ref_track_data is not None:
            self.ref_track_data = ref_track_data
        self.__validate()

    def __validate(self) -> None:
        validate_choice("sort_by", self.sort_by, SCAIL2COLOREDMASK_SORT_BIES)
        validate_bool("replacement_mode", self.replacement_mode)
