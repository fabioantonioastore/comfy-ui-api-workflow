from typing import Literal, get_args
from ...base import BaseNode
from ...validation import (
    validate_range,
    validate_choice,
)


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


MiniMaxH3ReferenceToVideo_Ref_Image_Sizes = Literal["match", "max"]


MINIMAXH3REFERENCETOVIDEO_REF_IMAGE_SIZES = frozenset(
    get_args(MiniMaxH3ReferenceToVideo_Ref_Image_Sizes)
)


class MiniMaxH3ReferenceToVideo(BaseNode):
    __slots__ = (
        "clip",
        "prompt",
        "width",
        "height",
        "length",
        "ref_image_size",
        "vae",
        "audio_vae",
        "ref_images_ref_image_0",
        "ref_images_ref_image_1",
        "ref_images_ref_image_2",
        "ref_images_ref_image_3",
        "ref_images_ref_image_4",
        "ref_images_ref_image_5",
        "ref_images_ref_image_6",
        "ref_images_ref_image_7",
        "ref_images_ref_image_8",
        "ref_videos_ref_video_0",
        "ref_videos_ref_video_1",
        "ref_videos_ref_video_2",
        "ref_video_audios_ref_video_audio_0",
        "ref_video_audios_ref_video_audio_1",
        "ref_video_audios_ref_video_audio_2",
        "ref_audios_ref_audio_0",
        "ref_audios_ref_audio_1",
        "ref_audios_ref_audio_2",
    )
    CATEGORY = "model/conditioning/minimax"
    OUTPUTS = {"positive": 0, "LATENT": 1}
    RETURN_TYPES = (
        "CONDITIONING",
        "LATENT",
    )
    INPUT_NAMES = {
        "ref_images_ref_image_0": "ref_images.ref_image_0",
        "ref_images_ref_image_1": "ref_images.ref_image_1",
        "ref_images_ref_image_2": "ref_images.ref_image_2",
        "ref_images_ref_image_3": "ref_images.ref_image_3",
        "ref_images_ref_image_4": "ref_images.ref_image_4",
        "ref_images_ref_image_5": "ref_images.ref_image_5",
        "ref_images_ref_image_6": "ref_images.ref_image_6",
        "ref_images_ref_image_7": "ref_images.ref_image_7",
        "ref_images_ref_image_8": "ref_images.ref_image_8",
        "ref_videos_ref_video_0": "ref_videos.ref_video_0",
        "ref_videos_ref_video_1": "ref_videos.ref_video_1",
        "ref_videos_ref_video_2": "ref_videos.ref_video_2",
        "ref_video_audios_ref_video_audio_0": "ref_video_audios.ref_video_audio_0",
        "ref_video_audios_ref_video_audio_1": "ref_video_audios.ref_video_audio_1",
        "ref_video_audios_ref_video_audio_2": "ref_video_audios.ref_video_audio_2",
        "ref_audios_ref_audio_0": "ref_audios.ref_audio_0",
        "ref_audios_ref_audio_1": "ref_audios.ref_audio_1",
        "ref_audios_ref_audio_2": "ref_audios.ref_audio_2",
    }
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "prompt": "STRING",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "ref_image_size": "COMBO",
        "vae": "VAE",
        "audio_vae": "VAE",
        "ref_images_ref_image_0": "IMAGE",
        "ref_images_ref_image_1": "IMAGE",
        "ref_images_ref_image_2": "IMAGE",
        "ref_images_ref_image_3": "IMAGE",
        "ref_images_ref_image_4": "IMAGE",
        "ref_images_ref_image_5": "IMAGE",
        "ref_images_ref_image_6": "IMAGE",
        "ref_images_ref_image_7": "IMAGE",
        "ref_images_ref_image_8": "IMAGE",
        "ref_videos_ref_video_0": "IMAGE",
        "ref_videos_ref_video_1": "IMAGE",
        "ref_videos_ref_video_2": "IMAGE",
        "ref_video_audios_ref_video_audio_0": "AUDIO",
        "ref_video_audios_ref_video_audio_1": "AUDIO",
        "ref_video_audios_ref_video_audio_2": "AUDIO",
        "ref_audios_ref_audio_0": "AUDIO",
        "ref_audios_ref_audio_1": "AUDIO",
        "ref_audios_ref_audio_2": "AUDIO",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        prompt: str | tuple[str, int],
        width: int | tuple[str, int] = 1344,
        height: int | tuple[str, int] = 768,
        length: int | tuple[str, int] = 124,
        ref_image_size: (
            MiniMaxH3ReferenceToVideo_Ref_Image_Sizes | tuple[str, int]
        ) = "match",
        vae: tuple[str, int] | None = None,
        audio_vae: tuple[str, int] | None = None,
        ref_images_ref_image_0: tuple[str, int] | None = None,
        ref_images_ref_image_1: tuple[str, int] | None = None,
        ref_images_ref_image_2: tuple[str, int] | None = None,
        ref_images_ref_image_3: tuple[str, int] | None = None,
        ref_images_ref_image_4: tuple[str, int] | None = None,
        ref_images_ref_image_5: tuple[str, int] | None = None,
        ref_images_ref_image_6: tuple[str, int] | None = None,
        ref_images_ref_image_7: tuple[str, int] | None = None,
        ref_images_ref_image_8: tuple[str, int] | None = None,
        ref_videos_ref_video_0: tuple[str, int] | None = None,
        ref_videos_ref_video_1: tuple[str, int] | None = None,
        ref_videos_ref_video_2: tuple[str, int] | None = None,
        ref_video_audios_ref_video_audio_0: tuple[str, int] | None = None,
        ref_video_audios_ref_video_audio_1: tuple[str, int] | None = None,
        ref_video_audios_ref_video_audio_2: tuple[str, int] | None = None,
        ref_audios_ref_audio_0: tuple[str, int] | None = None,
        ref_audios_ref_audio_1: tuple[str, int] | None = None,
        ref_audios_ref_audio_2: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.prompt = prompt
        self.width = width
        self.height = height
        self.length = length
        self.ref_image_size = ref_image_size
        if vae is not None:
            self.vae = vae
        if audio_vae is not None:
            self.audio_vae = audio_vae
        if ref_images_ref_image_0 is not None:
            self.ref_images_ref_image_0 = ref_images_ref_image_0
        if ref_images_ref_image_1 is not None:
            self.ref_images_ref_image_1 = ref_images_ref_image_1
        if ref_images_ref_image_2 is not None:
            self.ref_images_ref_image_2 = ref_images_ref_image_2
        if ref_images_ref_image_3 is not None:
            self.ref_images_ref_image_3 = ref_images_ref_image_3
        if ref_images_ref_image_4 is not None:
            self.ref_images_ref_image_4 = ref_images_ref_image_4
        if ref_images_ref_image_5 is not None:
            self.ref_images_ref_image_5 = ref_images_ref_image_5
        if ref_images_ref_image_6 is not None:
            self.ref_images_ref_image_6 = ref_images_ref_image_6
        if ref_images_ref_image_7 is not None:
            self.ref_images_ref_image_7 = ref_images_ref_image_7
        if ref_images_ref_image_8 is not None:
            self.ref_images_ref_image_8 = ref_images_ref_image_8
        if ref_videos_ref_video_0 is not None:
            self.ref_videos_ref_video_0 = ref_videos_ref_video_0
        if ref_videos_ref_video_1 is not None:
            self.ref_videos_ref_video_1 = ref_videos_ref_video_1
        if ref_videos_ref_video_2 is not None:
            self.ref_videos_ref_video_2 = ref_videos_ref_video_2
        if ref_video_audios_ref_video_audio_0 is not None:
            self.ref_video_audios_ref_video_audio_0 = ref_video_audios_ref_video_audio_0
        if ref_video_audios_ref_video_audio_1 is not None:
            self.ref_video_audios_ref_video_audio_1 = ref_video_audios_ref_video_audio_1
        if ref_video_audios_ref_video_audio_2 is not None:
            self.ref_video_audios_ref_video_audio_2 = ref_video_audios_ref_video_audio_2
        if ref_audios_ref_audio_0 is not None:
            self.ref_audios_ref_audio_0 = ref_audios_ref_audio_0
        if ref_audios_ref_audio_1 is not None:
            self.ref_audios_ref_audio_1 = ref_audios_ref_audio_1
        if ref_audios_ref_audio_2 is not None:
            self.ref_audios_ref_audio_2 = ref_audios_ref_audio_2
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 32, 16384)
        validate_range("height", self.height, 32, 16384)
        validate_range("length", self.length, 5, 3600)
        validate_choice(
            "ref_image_size",
            self.ref_image_size,
            MINIMAXH3REFERENCETOVIDEO_REF_IMAGE_SIZES,
        )
