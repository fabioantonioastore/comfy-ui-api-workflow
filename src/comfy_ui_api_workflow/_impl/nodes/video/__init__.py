from typing import Any, Literal, get_args

from ...errors.node import InvalidNodeParameterValue
from ..base import BaseNode
from ..validation import (
    validate_bool,
    validate_choice,
    validate_not_empty,
    validate_range,
)
from ...lazy import submodules_of


class LoadVideoDataSetFromFolder(BaseNode):
    __slots__ = ("folder",)
    CATEGORY = "video"
    OUTPUTS = {"videos": 0}
    RETURN_TYPES = ("VIDEO",)
    INPUT_SOCKET_TYPES = {"folder": "COMBO"}

    def __init__(self, node_id: str, folder: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.folder = folder
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("folder", self.folder)


class LoadVideoTextDataSetFromFolder(BaseNode):
    __slots__ = ("folder",)
    CATEGORY = "video"
    OUTPUTS = {"videos": 0, "texts": 1}
    RETURN_TYPES = ("VIDEO", "STRING")
    INPUT_SOCKET_TYPES = {"folder": "COMBO"}

    def __init__(self, node_id: str, folder: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.folder = folder
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("folder", self.folder)


VideoFrameSample_Strategies = Literal["uniform", "head", "tail", "random"]
VIDEOFRAMESAMPLE_STRATEGIES = frozenset(get_args(VideoFrameSample_Strategies))


class VideoFrameSample(BaseNode):
    __slots__ = ("video", "num_frames", "strategy", "seed")
    CATEGORY = "video"
    OUTPUTS = {"video": 0}
    RETURN_TYPES = ("VIDEO",)
    INPUT_SOCKET_TYPES = {
        "video": "VIDEO",
        "num_frames": "INT",
        "strategy": "COMBO",
        "seed": "INT",
    }

    def __init__(
        self,
        node_id: str,
        video: tuple[str, int],
        num_frames: int | tuple[str, int] = 16,
        strategy: VideoFrameSample_Strategies | tuple[str, int] = "uniform",
        seed: int | tuple[str, int] = 0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.video = video
        self.num_frames = num_frames
        self.strategy = strategy
        self.seed = seed
        self.__validate()

    def __validate(self) -> None:
        validate_range("num_frames", self.num_frames, 1, 9999)
        validate_choice("strategy", self.strategy, VIDEOFRAMESAMPLE_STRATEGIES)
        validate_range("seed", self.seed, 0, 18_446_744_073_709_551_615)


SaveWEBM_Codecs = Literal["vp9", "av1"]
SAVEWEBM_CODECS = frozenset(get_args(SaveWEBM_Codecs))


class SaveWEBM(BaseNode):
    __slots__ = ("images", "codec", "filename_prefix", "fps", "crf")
    CATEGORY = "video"
    OUTPUTS = {"images": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "images": "IMAGE",
        "filename_prefix": "STRING",
        "codec": "COMBO",
        "fps": "FLOAT",
        "crf": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        codec: SaveWEBM_Codecs | tuple[str, int],
        filename_prefix: str | tuple[str, int] = "ComfyUI",
        fps: float | tuple[str, int] = 24.0,
        crf: float | tuple[str, int] = 32.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.codec = codec
        self.filename_prefix = filename_prefix
        self.fps = fps
        self.crf = crf
        self.__validate()

    def __validate(self) -> None:
        validate_choice("codec", self.codec, SAVEWEBM_CODECS)
        validate_range("fps", self.fps, 0.01, 1000.0)
        validate_range("crf", self.crf, 0, 63.0)


CreateVideo_Bit_Depths = Literal["auto", 8, 10]
CreateVideo_Color_Spaces = Literal["sRGB", "HDR", "HDR PQ"]
CreateVideo_Codecs = Literal["none", "auto", "h264", "av1"]
CREATEVIDEO_BIT_DEPTHS = frozenset(get_args(CreateVideo_Bit_Depths))
CREATEVIDEO_COLOR_SPACES = frozenset(get_args(CreateVideo_Color_Spaces))
CREATEVIDEO_CODECS = frozenset(get_args(CreateVideo_Codecs))


class CreateVideo(BaseNode):
    __slots__ = ("images", "fps", "audio", "bit_depth", "color_space", "codec")
    CATEGORY = "video"
    OUTPUTS = {"VIDEO": 0}
    RETURN_TYPES = ("VIDEO",)
    INPUT_SOCKET_TYPES = {
        "images": "IMAGE",
        "fps": "FLOAT",
        "audio": "AUDIO",
        "bit_depth": "COMBO",
        "color_space": "COMBO",
        "codec": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        fps: float | tuple[str, int] = 30.0,
        audio: tuple[str, int] | None = None,
        bit_depth: CreateVideo_Bit_Depths | tuple[str, int] = "auto",
        color_space: CreateVideo_Color_Spaces | tuple[str, int] = "sRGB",
        codec: CreateVideo_Codecs | tuple[str, int] = "none",
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.fps = fps
        if audio is not None:
            self.audio = audio
        self.bit_depth = bit_depth
        self.color_space = color_space
        self.codec = codec
        self.__validate()

    def __validate(self) -> None:
        validate_range("fps", self.fps, 1.0, 120.0)
        validate_choice("bit_depth", self.bit_depth, CREATEVIDEO_BIT_DEPTHS)
        validate_choice("color_space", self.color_space, CREATEVIDEO_COLOR_SPACES)
        validate_choice("codec", self.codec, CREATEVIDEO_CODECS)


class GetVideoComponents(BaseNode):
    __slots__ = ("video",)
    CATEGORY = "video"
    OUTPUTS = {"images": 0, "audio": 1, "fps": 2, "bit_depth": 3, "color_space": 4}
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "COMBO", "COMBO")
    INPUT_SOCKET_TYPES = {"video": "VIDEO"}

    def __init__(self, node_id: str, video: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.video = video


class LoadVideo(BaseNode):
    __slots__ = ("file",)
    CATEGORY = "video"
    OUTPUTS = {"VIDEO": 0}
    RETURN_TYPES = ("VIDEO",)
    INPUT_SOCKET_TYPES = {"file": "COMBO"}

    def __init__(self, node_id: str, file: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.file = file
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("file", self.file)


class VideoSlice(BaseNode):
    __slots__ = ("video", "start_time", "duration", "strict_duration")
    CATEGORY = "video"
    CLASS_TYPE = "Video Slice"
    OUTPUTS = {"VIDEO": 0}
    RETURN_TYPES = ("VIDEO",)
    INPUT_SOCKET_TYPES = {
        "video": "VIDEO",
        "start_time": "FLOAT",
        "duration": "FLOAT",
        "strict_duration": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        video: tuple[str, int],
        start_time: float | tuple[str, int] = 0.0,
        duration: float | tuple[str, int] = 0.0,
        strict_duration: bool | tuple[str, int] = False,
    ) -> None:
        super().__init__(node_id=node_id)
        self.video = video
        self.start_time = start_time
        self.duration = duration
        self.strict_duration = strict_duration
        self.__validate()

    def __validate(self) -> None:
        validate_range("start_time", self.start_time, -100000.0, 100000.0)
        validate_range("duration", self.duration, 0.0)
        validate_bool("strict_duration", self.strict_duration)


class VideoTrim(BaseNode):
    __slots__ = ("video", "trim", "strict_duration")
    CATEGORY = "video"
    OUTPUTS = {"VIDEO": 0}
    RETURN_TYPES = ("VIDEO",)
    INPUT_SOCKET_TYPES = {
        "video": "VIDEO",
        "trim": "VIDEO_EDIT",
        "strict_duration": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        video: tuple[str, int],
        trim: Any = None,
        strict_duration: bool | tuple[str, int] = False,
    ) -> None:
        super().__init__(node_id=node_id)
        self.video = video
        self.trim = (
            {"trim": {"start_time": 0.0, "duration": 0.0}} if trim is None else trim
        )
        self.strict_duration = strict_duration
        self.__validate()

    def __validate(self) -> None:
        validate_bool("strict_duration", self.strict_duration)


class VideoCrop(BaseNode):
    __slots__ = ("video", "crop")
    CATEGORY = "video"
    OUTPUTS = {"VIDEO": 0}
    RETURN_TYPES = ("VIDEO",)
    INPUT_SOCKET_TYPES = {"video": "VIDEO", "crop": "VIDEO_EDIT"}

    def __init__(self, node_id: str, video: tuple[str, int], crop: Any = None) -> None:
        super().__init__(node_id=node_id)
        self.video = video
        self.crop = (
            {"crop": {"x": 0, "y": 0, "width": 0, "height": 0}}
            if crop is None
            else crop
        )


class FrameInterpolate(BaseNode):
    __slots__ = ("interp_model", "images", "multiplier")
    CATEGORY = "video"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "interp_model": "INTERP_MODEL",
        "images": "IMAGE",
        "multiplier": "INT",
    }

    def __init__(
        self,
        node_id: str,
        interp_model: tuple[str, int],
        images: tuple[str, int],
        multiplier: int | tuple[str, int] = 2,
    ) -> None:
        super().__init__(node_id=node_id)
        self.interp_model = interp_model
        self.images = images
        self.multiplier = multiplier
        self.__validate()

    def __validate(self) -> None:
        validate_range("multiplier", self.multiplier, 2, 16)


SaveVideo_Formats = Literal["auto", "mp4", "mkv", "webm"]
SaveVideo_Codecs = Literal["auto", "h264", "av1"]
SaveVideo_Encodings = Literal["auto", "re-encode"]
SAVEVIDEO_FORMATS = frozenset(get_args(SaveVideo_Formats))
SAVEVIDEO_CODECS = frozenset(get_args(SaveVideo_Codecs))
SAVEVIDEO_ENCODINGS = frozenset(get_args(SaveVideo_Encodings))
# The codecs each format offers and the crf range of each codec, as in object_info.
SAVEVIDEO_FORMAT_CODECS = {
    "auto": SAVEVIDEO_CODECS,
    "mp4": SAVEVIDEO_CODECS,
    "mkv": SAVEVIDEO_CODECS,
    "webm": frozenset({"auto", "av1"}),
}
SAVEVIDEO_CRF_RANGES = {"h264": (0.0, 51.0), "av1": (0.0, 63.0)}


def _validate_codec_branch(prefix: str, codec: Any, encoding: Any, crf: Any) -> None:
    # Shared by the format.codec branch and the flat codec group of SaveVideo:
    # encoding only exists for h264 and av1, and crf only for "re-encode".
    encoding_name = f"{prefix}_encoding"
    crf_name = f"{encoding_name}_crf"
    if encoding is not None and codec not in SAVEVIDEO_CRF_RANGES:
        raise InvalidNodeParameterValue(
            f"{encoding_name} only applies when {prefix} is h264 or av1, not {codec!r}"
        )
    if encoding is not None:
        validate_choice(encoding_name, encoding, SAVEVIDEO_ENCODINGS)
    if encoding == "re-encode":
        if crf is None:
            raise InvalidNodeParameterValue(
                f"{crf_name} is required when {encoding_name} is 're-encode'"
            )
        validate_range(crf_name, crf, *SAVEVIDEO_CRF_RANGES[codec])
    elif crf is not None:
        raise InvalidNodeParameterValue(
            f"{crf_name} only applies when {encoding_name} is 're-encode'"
        )


# The dynamic combo "format" opens "format.codec", then "format.codec.encoding",
# then "format.codec.encoding.crf"; each level is an attribute sent with its
# dotted name. The flat "codec" group follows the same tree.
class SaveVideo(BaseNode):
    __slots__ = (
        "video",
        "filename_prefix",
        "format",
        "format_codec",
        "format_codec_encoding",
        "format_codec_encoding_crf",
        "codec",
        "codec_encoding",
        "codec_encoding_crf",
    )
    CATEGORY = "video"
    OUTPUTS = {"video": 0}
    RETURN_TYPES = ("VIDEO",)
    INPUT_NAMES = {
        "format_codec": "format.codec",
        "format_codec_encoding": "format.codec.encoding",
        "format_codec_encoding_crf": "format.codec.encoding.crf",
        "codec_encoding": "codec.encoding",
        "codec_encoding_crf": "codec.encoding.crf",
    }
    INPUT_SOCKET_TYPES = {
        "video": "VIDEO",
        "filename_prefix": "STRING",
        "format": "COMFY_DYNAMICCOMBO_V3",
        "format_codec": "COMFY_DYNAMICCOMBO_V3",
        "format_codec_encoding": "COMFY_DYNAMICCOMBO_V3",
        "format_codec_encoding_crf": "FLOAT",
        "codec": "COMFY_DYNAMICCOMBO_V3",
        "codec_encoding": "COMFY_DYNAMICCOMBO_V3",
        "codec_encoding_crf": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        video: tuple[str, int],
        filename_prefix: str | tuple[str, int] = "video/ComfyUI",
        format: SaveVideo_Formats = "auto",
        format_codec: SaveVideo_Codecs = "auto",
        format_codec_encoding: SaveVideo_Encodings | None = None,
        format_codec_encoding_crf: float | tuple[str, int] | None = None,
        codec: SaveVideo_Codecs | None = None,
        codec_encoding: SaveVideo_Encodings | None = None,
        codec_encoding_crf: float | tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.video = video
        self.filename_prefix = filename_prefix
        self.format = format
        self.format_codec = format_codec
        if format_codec_encoding is not None:
            self.format_codec_encoding = format_codec_encoding
        if format_codec_encoding_crf is not None:
            self.format_codec_encoding_crf = format_codec_encoding_crf
        if codec is not None:
            self.codec = codec
        if codec_encoding is not None:
            self.codec_encoding = codec_encoding
        if codec_encoding_crf is not None:
            self.codec_encoding_crf = codec_encoding_crf
        self.__validate()

    def __validate(self) -> None:
        validate_choice("format", self.format, SAVEVIDEO_FORMATS)
        codecs = SAVEVIDEO_FORMAT_CODECS[self.format]
        if self.format_codec not in codecs:
            raise InvalidNodeParameterValue(
                f"format_codec {self.format_codec!r} is not available for format "
                f"{self.format!r}; use one of {sorted(codecs)}"
            )
        _validate_codec_branch(
            "format_codec",
            self.format_codec,
            getattr(self, "format_codec_encoding", None),
            getattr(self, "format_codec_encoding_crf", None),
        )
        codec = getattr(self, "codec", None)
        if codec is not None:
            validate_choice("codec", codec, SAVEVIDEO_CODECS)
        _validate_codec_branch(
            "codec",
            codec,
            getattr(self, "codec_encoding", None),
            getattr(self, "codec_encoding_crf", None),
        )


__getattr__ = submodules_of(__name__)
