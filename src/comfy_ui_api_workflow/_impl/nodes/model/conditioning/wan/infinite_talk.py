from typing import Literal

from ....base import BaseNode
from ....dynamic import Combo, DynamicInputs, Link, validate_dynamic
from ....validation import validate_range

WanInfiniteTalkToVideo_Modes = Literal["single_speaker", "two_speakers"]


class WanInfiniteTalkToVideo(DynamicInputs, BaseNode):
    __slots__ = (
        "mode",
        "mode_audio_encoder_output_2",
        "mode_mask_1",
        "mode_mask_2",
        "model",
        "model_patch",
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "audio_encoder_output_1",
        "motion_frame_count",
        "audio_scale",
        "clip_vision_output",
        "start_image",
        "previous_frames",
    )
    CATEGORY = "model/conditioning/wan/infinite talk"
    OUTPUTS = {"model": 0, "positive": 1, "negative": 2, "latent": 3, "trim_image": 4}
    RETURN_TYPES = (
        "MODEL",
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
        "INT",
    )
    INPUT_NAMES = {
        "mode_audio_encoder_output_2": "mode.audio_encoder_output_2",
        "mode_mask_1": "mode.mask_1",
        "mode_mask_2": "mode.mask_2",
    }
    INPUT_SOCKET_TYPES = {
        "mode": "COMFY_DYNAMICCOMBO_V3",
        "mode_audio_encoder_output_2": "AUDIO_ENCODER_OUTPUT",
        "mode_mask_1": "MASK",
        "mode_mask_2": "MASK",
        "model": "MODEL",
        "model_patch": "MODEL_PATCH",
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "audio_encoder_output_1": "AUDIO_ENCODER_OUTPUT",
        "motion_frame_count": "INT",
        "audio_scale": "FLOAT",
        "clip_vision_output": "CLIP_VISION_OUTPUT",
        "start_image": "IMAGE",
        "previous_frames": "IMAGE",
    }
    DYNAMIC_INPUTS = {
        "mode": Combo(
            {
                "single_speaker": {},
                "two_speakers": {
                    "mode_audio_encoder_output_2": Link(required=False),
                    "mode_mask_1": Link(required=False),
                    "mode_mask_2": Link(required=False),
                },
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        model_patch: tuple[str, int],
        positive: tuple[str, int],
        negative: tuple[str, int],
        vae: tuple[str, int],
        audio_encoder_output_1: tuple[str, int],
        mode: WanInfiniteTalkToVideo_Modes = "single_speaker",
        mode_audio_encoder_output_2: tuple[str, int] | None = None,
        mode_mask_1: tuple[str, int] | None = None,
        mode_mask_2: tuple[str, int] | None = None,
        width: int | tuple[str, int] = 832,
        height: int | tuple[str, int] = 480,
        length: int | tuple[str, int] = 81,
        motion_frame_count: int | tuple[str, int] = 9,
        audio_scale: float | tuple[str, int] = 1.0,
        clip_vision_output: tuple[str, int] | None = None,
        start_image: tuple[str, int] | None = None,
        previous_frames: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mode = mode
        if mode_audio_encoder_output_2 is not None:
            self.mode_audio_encoder_output_2 = mode_audio_encoder_output_2
        if mode_mask_1 is not None:
            self.mode_mask_1 = mode_mask_1
        if mode_mask_2 is not None:
            self.mode_mask_2 = mode_mask_2
        self.model = model
        self.model_patch = model_patch
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.width = width
        self.height = height
        self.length = length
        self.audio_encoder_output_1 = audio_encoder_output_1
        self.motion_frame_count = motion_frame_count
        self.audio_scale = audio_scale
        if clip_vision_output is not None:
            self.clip_vision_output = clip_vision_output
        if start_image is not None:
            self.start_image = start_image
        if previous_frames is not None:
            self.previous_frames = previous_frames
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16384)
        validate_range("height", self.height, 16, 16384)
        validate_range("length", self.length, 1, 16384)
        validate_range("motion_frame_count", self.motion_frame_count, 1, 33)
        validate_range("audio_scale", self.audio_scale, -10.0, 10.0)
        validate_dynamic(self)
