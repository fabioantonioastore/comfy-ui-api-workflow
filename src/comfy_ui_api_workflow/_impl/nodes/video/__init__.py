from typing import Any, Literal, get_args

from ...errors.node import InvalidNodeParameterValue
from ..base import BaseNode
from ..validation import (
    validate_bool,
    validate_choice,
    validate_not_empty,
    validate_range,
)


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


ConcatenateVideo_Codecs = Literal["auto", "h264", "av1"]


CONCATENATEVIDEO_CODECS = frozenset(get_args(ConcatenateVideo_Codecs))


class ConcatenateVideo(BaseNode):
    __slots__ = (
        "videos_video0",
        "videos_video1",
        "videos_video2",
        "videos_video3",
        "videos_video4",
        "videos_video5",
        "videos_video6",
        "videos_video7",
        "videos_video8",
        "videos_video9",
        "videos_video10",
        "videos_video11",
        "videos_video12",
        "videos_video13",
        "videos_video14",
        "videos_video15",
        "videos_video16",
        "videos_video17",
        "videos_video18",
        "videos_video19",
        "videos_video20",
        "videos_video21",
        "videos_video22",
        "videos_video23",
        "videos_video24",
        "videos_video25",
        "videos_video26",
        "videos_video27",
        "videos_video28",
        "videos_video29",
        "videos_video30",
        "videos_video31",
        "videos_video32",
        "videos_video33",
        "videos_video34",
        "videos_video35",
        "videos_video36",
        "videos_video37",
        "videos_video38",
        "videos_video39",
        "videos_video40",
        "videos_video41",
        "videos_video42",
        "videos_video43",
        "videos_video44",
        "videos_video45",
        "videos_video46",
        "videos_video47",
        "videos_video48",
        "videos_video49",
        "videos_video50",
        "videos_video51",
        "videos_video52",
        "videos_video53",
        "videos_video54",
        "videos_video55",
        "videos_video56",
        "videos_video57",
        "videos_video58",
        "videos_video59",
        "videos_video60",
        "videos_video61",
        "videos_video62",
        "videos_video63",
        "videos_video64",
        "videos_video65",
        "videos_video66",
        "videos_video67",
        "videos_video68",
        "videos_video69",
        "videos_video70",
        "videos_video71",
        "videos_video72",
        "videos_video73",
        "videos_video74",
        "videos_video75",
        "videos_video76",
        "videos_video77",
        "videos_video78",
        "videos_video79",
        "videos_video80",
        "videos_video81",
        "videos_video82",
        "videos_video83",
        "videos_video84",
        "videos_video85",
        "videos_video86",
        "videos_video87",
        "videos_video88",
        "videos_video89",
        "videos_video90",
        "videos_video91",
        "videos_video92",
        "videos_video93",
        "videos_video94",
        "videos_video95",
        "videos_video96",
        "videos_video97",
        "videos_video98",
        "videos_video99",
        "codec",
        "complete_audio",
    )
    CATEGORY = "video"
    OUTPUTS = {"VIDEO": 0}
    RETURN_TYPES = ("VIDEO",)
    INPUT_NAMES = {
        "videos_video0": "videos.video0",
        "videos_video1": "videos.video1",
        "videos_video2": "videos.video2",
        "videos_video3": "videos.video3",
        "videos_video4": "videos.video4",
        "videos_video5": "videos.video5",
        "videos_video6": "videos.video6",
        "videos_video7": "videos.video7",
        "videos_video8": "videos.video8",
        "videos_video9": "videos.video9",
        "videos_video10": "videos.video10",
        "videos_video11": "videos.video11",
        "videos_video12": "videos.video12",
        "videos_video13": "videos.video13",
        "videos_video14": "videos.video14",
        "videos_video15": "videos.video15",
        "videos_video16": "videos.video16",
        "videos_video17": "videos.video17",
        "videos_video18": "videos.video18",
        "videos_video19": "videos.video19",
        "videos_video20": "videos.video20",
        "videos_video21": "videos.video21",
        "videos_video22": "videos.video22",
        "videos_video23": "videos.video23",
        "videos_video24": "videos.video24",
        "videos_video25": "videos.video25",
        "videos_video26": "videos.video26",
        "videos_video27": "videos.video27",
        "videos_video28": "videos.video28",
        "videos_video29": "videos.video29",
        "videos_video30": "videos.video30",
        "videos_video31": "videos.video31",
        "videos_video32": "videos.video32",
        "videos_video33": "videos.video33",
        "videos_video34": "videos.video34",
        "videos_video35": "videos.video35",
        "videos_video36": "videos.video36",
        "videos_video37": "videos.video37",
        "videos_video38": "videos.video38",
        "videos_video39": "videos.video39",
        "videos_video40": "videos.video40",
        "videos_video41": "videos.video41",
        "videos_video42": "videos.video42",
        "videos_video43": "videos.video43",
        "videos_video44": "videos.video44",
        "videos_video45": "videos.video45",
        "videos_video46": "videos.video46",
        "videos_video47": "videos.video47",
        "videos_video48": "videos.video48",
        "videos_video49": "videos.video49",
        "videos_video50": "videos.video50",
        "videos_video51": "videos.video51",
        "videos_video52": "videos.video52",
        "videos_video53": "videos.video53",
        "videos_video54": "videos.video54",
        "videos_video55": "videos.video55",
        "videos_video56": "videos.video56",
        "videos_video57": "videos.video57",
        "videos_video58": "videos.video58",
        "videos_video59": "videos.video59",
        "videos_video60": "videos.video60",
        "videos_video61": "videos.video61",
        "videos_video62": "videos.video62",
        "videos_video63": "videos.video63",
        "videos_video64": "videos.video64",
        "videos_video65": "videos.video65",
        "videos_video66": "videos.video66",
        "videos_video67": "videos.video67",
        "videos_video68": "videos.video68",
        "videos_video69": "videos.video69",
        "videos_video70": "videos.video70",
        "videos_video71": "videos.video71",
        "videos_video72": "videos.video72",
        "videos_video73": "videos.video73",
        "videos_video74": "videos.video74",
        "videos_video75": "videos.video75",
        "videos_video76": "videos.video76",
        "videos_video77": "videos.video77",
        "videos_video78": "videos.video78",
        "videos_video79": "videos.video79",
        "videos_video80": "videos.video80",
        "videos_video81": "videos.video81",
        "videos_video82": "videos.video82",
        "videos_video83": "videos.video83",
        "videos_video84": "videos.video84",
        "videos_video85": "videos.video85",
        "videos_video86": "videos.video86",
        "videos_video87": "videos.video87",
        "videos_video88": "videos.video88",
        "videos_video89": "videos.video89",
        "videos_video90": "videos.video90",
        "videos_video91": "videos.video91",
        "videos_video92": "videos.video92",
        "videos_video93": "videos.video93",
        "videos_video94": "videos.video94",
        "videos_video95": "videos.video95",
        "videos_video96": "videos.video96",
        "videos_video97": "videos.video97",
        "videos_video98": "videos.video98",
        "videos_video99": "videos.video99",
    }
    INPUT_SOCKET_TYPES = {
        "videos_video0": "VIDEO",
        "videos_video1": "VIDEO",
        "videos_video2": "VIDEO",
        "videos_video3": "VIDEO",
        "videos_video4": "VIDEO",
        "videos_video5": "VIDEO",
        "videos_video6": "VIDEO",
        "videos_video7": "VIDEO",
        "videos_video8": "VIDEO",
        "videos_video9": "VIDEO",
        "videos_video10": "VIDEO",
        "videos_video11": "VIDEO",
        "videos_video12": "VIDEO",
        "videos_video13": "VIDEO",
        "videos_video14": "VIDEO",
        "videos_video15": "VIDEO",
        "videos_video16": "VIDEO",
        "videos_video17": "VIDEO",
        "videos_video18": "VIDEO",
        "videos_video19": "VIDEO",
        "videos_video20": "VIDEO",
        "videos_video21": "VIDEO",
        "videos_video22": "VIDEO",
        "videos_video23": "VIDEO",
        "videos_video24": "VIDEO",
        "videos_video25": "VIDEO",
        "videos_video26": "VIDEO",
        "videos_video27": "VIDEO",
        "videos_video28": "VIDEO",
        "videos_video29": "VIDEO",
        "videos_video30": "VIDEO",
        "videos_video31": "VIDEO",
        "videos_video32": "VIDEO",
        "videos_video33": "VIDEO",
        "videos_video34": "VIDEO",
        "videos_video35": "VIDEO",
        "videos_video36": "VIDEO",
        "videos_video37": "VIDEO",
        "videos_video38": "VIDEO",
        "videos_video39": "VIDEO",
        "videos_video40": "VIDEO",
        "videos_video41": "VIDEO",
        "videos_video42": "VIDEO",
        "videos_video43": "VIDEO",
        "videos_video44": "VIDEO",
        "videos_video45": "VIDEO",
        "videos_video46": "VIDEO",
        "videos_video47": "VIDEO",
        "videos_video48": "VIDEO",
        "videos_video49": "VIDEO",
        "videos_video50": "VIDEO",
        "videos_video51": "VIDEO",
        "videos_video52": "VIDEO",
        "videos_video53": "VIDEO",
        "videos_video54": "VIDEO",
        "videos_video55": "VIDEO",
        "videos_video56": "VIDEO",
        "videos_video57": "VIDEO",
        "videos_video58": "VIDEO",
        "videos_video59": "VIDEO",
        "videos_video60": "VIDEO",
        "videos_video61": "VIDEO",
        "videos_video62": "VIDEO",
        "videos_video63": "VIDEO",
        "videos_video64": "VIDEO",
        "videos_video65": "VIDEO",
        "videos_video66": "VIDEO",
        "videos_video67": "VIDEO",
        "videos_video68": "VIDEO",
        "videos_video69": "VIDEO",
        "videos_video70": "VIDEO",
        "videos_video71": "VIDEO",
        "videos_video72": "VIDEO",
        "videos_video73": "VIDEO",
        "videos_video74": "VIDEO",
        "videos_video75": "VIDEO",
        "videos_video76": "VIDEO",
        "videos_video77": "VIDEO",
        "videos_video78": "VIDEO",
        "videos_video79": "VIDEO",
        "videos_video80": "VIDEO",
        "videos_video81": "VIDEO",
        "videos_video82": "VIDEO",
        "videos_video83": "VIDEO",
        "videos_video84": "VIDEO",
        "videos_video85": "VIDEO",
        "videos_video86": "VIDEO",
        "videos_video87": "VIDEO",
        "videos_video88": "VIDEO",
        "videos_video89": "VIDEO",
        "videos_video90": "VIDEO",
        "videos_video91": "VIDEO",
        "videos_video92": "VIDEO",
        "videos_video93": "VIDEO",
        "videos_video94": "VIDEO",
        "videos_video95": "VIDEO",
        "videos_video96": "VIDEO",
        "videos_video97": "VIDEO",
        "videos_video98": "VIDEO",
        "videos_video99": "VIDEO",
        "codec": "COMBO",
        "complete_audio": "AUDIO",
    }

    def __init__(
        self,
        node_id: str,
        videos_video0: tuple[str, int],
        videos_video1: tuple[str, int] | None = None,
        videos_video2: tuple[str, int] | None = None,
        videos_video3: tuple[str, int] | None = None,
        videos_video4: tuple[str, int] | None = None,
        videos_video5: tuple[str, int] | None = None,
        videos_video6: tuple[str, int] | None = None,
        videos_video7: tuple[str, int] | None = None,
        videos_video8: tuple[str, int] | None = None,
        videos_video9: tuple[str, int] | None = None,
        videos_video10: tuple[str, int] | None = None,
        videos_video11: tuple[str, int] | None = None,
        videos_video12: tuple[str, int] | None = None,
        videos_video13: tuple[str, int] | None = None,
        videos_video14: tuple[str, int] | None = None,
        videos_video15: tuple[str, int] | None = None,
        videos_video16: tuple[str, int] | None = None,
        videos_video17: tuple[str, int] | None = None,
        videos_video18: tuple[str, int] | None = None,
        videos_video19: tuple[str, int] | None = None,
        videos_video20: tuple[str, int] | None = None,
        videos_video21: tuple[str, int] | None = None,
        videos_video22: tuple[str, int] | None = None,
        videos_video23: tuple[str, int] | None = None,
        videos_video24: tuple[str, int] | None = None,
        videos_video25: tuple[str, int] | None = None,
        videos_video26: tuple[str, int] | None = None,
        videos_video27: tuple[str, int] | None = None,
        videos_video28: tuple[str, int] | None = None,
        videos_video29: tuple[str, int] | None = None,
        videos_video30: tuple[str, int] | None = None,
        videos_video31: tuple[str, int] | None = None,
        videos_video32: tuple[str, int] | None = None,
        videos_video33: tuple[str, int] | None = None,
        videos_video34: tuple[str, int] | None = None,
        videos_video35: tuple[str, int] | None = None,
        videos_video36: tuple[str, int] | None = None,
        videos_video37: tuple[str, int] | None = None,
        videos_video38: tuple[str, int] | None = None,
        videos_video39: tuple[str, int] | None = None,
        videos_video40: tuple[str, int] | None = None,
        videos_video41: tuple[str, int] | None = None,
        videos_video42: tuple[str, int] | None = None,
        videos_video43: tuple[str, int] | None = None,
        videos_video44: tuple[str, int] | None = None,
        videos_video45: tuple[str, int] | None = None,
        videos_video46: tuple[str, int] | None = None,
        videos_video47: tuple[str, int] | None = None,
        videos_video48: tuple[str, int] | None = None,
        videos_video49: tuple[str, int] | None = None,
        videos_video50: tuple[str, int] | None = None,
        videos_video51: tuple[str, int] | None = None,
        videos_video52: tuple[str, int] | None = None,
        videos_video53: tuple[str, int] | None = None,
        videos_video54: tuple[str, int] | None = None,
        videos_video55: tuple[str, int] | None = None,
        videos_video56: tuple[str, int] | None = None,
        videos_video57: tuple[str, int] | None = None,
        videos_video58: tuple[str, int] | None = None,
        videos_video59: tuple[str, int] | None = None,
        videos_video60: tuple[str, int] | None = None,
        videos_video61: tuple[str, int] | None = None,
        videos_video62: tuple[str, int] | None = None,
        videos_video63: tuple[str, int] | None = None,
        videos_video64: tuple[str, int] | None = None,
        videos_video65: tuple[str, int] | None = None,
        videos_video66: tuple[str, int] | None = None,
        videos_video67: tuple[str, int] | None = None,
        videos_video68: tuple[str, int] | None = None,
        videos_video69: tuple[str, int] | None = None,
        videos_video70: tuple[str, int] | None = None,
        videos_video71: tuple[str, int] | None = None,
        videos_video72: tuple[str, int] | None = None,
        videos_video73: tuple[str, int] | None = None,
        videos_video74: tuple[str, int] | None = None,
        videos_video75: tuple[str, int] | None = None,
        videos_video76: tuple[str, int] | None = None,
        videos_video77: tuple[str, int] | None = None,
        videos_video78: tuple[str, int] | None = None,
        videos_video79: tuple[str, int] | None = None,
        videos_video80: tuple[str, int] | None = None,
        videos_video81: tuple[str, int] | None = None,
        videos_video82: tuple[str, int] | None = None,
        videos_video83: tuple[str, int] | None = None,
        videos_video84: tuple[str, int] | None = None,
        videos_video85: tuple[str, int] | None = None,
        videos_video86: tuple[str, int] | None = None,
        videos_video87: tuple[str, int] | None = None,
        videos_video88: tuple[str, int] | None = None,
        videos_video89: tuple[str, int] | None = None,
        videos_video90: tuple[str, int] | None = None,
        videos_video91: tuple[str, int] | None = None,
        videos_video92: tuple[str, int] | None = None,
        videos_video93: tuple[str, int] | None = None,
        videos_video94: tuple[str, int] | None = None,
        videos_video95: tuple[str, int] | None = None,
        videos_video96: tuple[str, int] | None = None,
        videos_video97: tuple[str, int] | None = None,
        videos_video98: tuple[str, int] | None = None,
        videos_video99: tuple[str, int] | None = None,
        codec: ConcatenateVideo_Codecs | tuple[str, int] = "auto",
        complete_audio: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.videos_video0 = videos_video0
        if videos_video1 is not None:
            self.videos_video1 = videos_video1
        if videos_video2 is not None:
            self.videos_video2 = videos_video2
        if videos_video3 is not None:
            self.videos_video3 = videos_video3
        if videos_video4 is not None:
            self.videos_video4 = videos_video4
        if videos_video5 is not None:
            self.videos_video5 = videos_video5
        if videos_video6 is not None:
            self.videos_video6 = videos_video6
        if videos_video7 is not None:
            self.videos_video7 = videos_video7
        if videos_video8 is not None:
            self.videos_video8 = videos_video8
        if videos_video9 is not None:
            self.videos_video9 = videos_video9
        if videos_video10 is not None:
            self.videos_video10 = videos_video10
        if videos_video11 is not None:
            self.videos_video11 = videos_video11
        if videos_video12 is not None:
            self.videos_video12 = videos_video12
        if videos_video13 is not None:
            self.videos_video13 = videos_video13
        if videos_video14 is not None:
            self.videos_video14 = videos_video14
        if videos_video15 is not None:
            self.videos_video15 = videos_video15
        if videos_video16 is not None:
            self.videos_video16 = videos_video16
        if videos_video17 is not None:
            self.videos_video17 = videos_video17
        if videos_video18 is not None:
            self.videos_video18 = videos_video18
        if videos_video19 is not None:
            self.videos_video19 = videos_video19
        if videos_video20 is not None:
            self.videos_video20 = videos_video20
        if videos_video21 is not None:
            self.videos_video21 = videos_video21
        if videos_video22 is not None:
            self.videos_video22 = videos_video22
        if videos_video23 is not None:
            self.videos_video23 = videos_video23
        if videos_video24 is not None:
            self.videos_video24 = videos_video24
        if videos_video25 is not None:
            self.videos_video25 = videos_video25
        if videos_video26 is not None:
            self.videos_video26 = videos_video26
        if videos_video27 is not None:
            self.videos_video27 = videos_video27
        if videos_video28 is not None:
            self.videos_video28 = videos_video28
        if videos_video29 is not None:
            self.videos_video29 = videos_video29
        if videos_video30 is not None:
            self.videos_video30 = videos_video30
        if videos_video31 is not None:
            self.videos_video31 = videos_video31
        if videos_video32 is not None:
            self.videos_video32 = videos_video32
        if videos_video33 is not None:
            self.videos_video33 = videos_video33
        if videos_video34 is not None:
            self.videos_video34 = videos_video34
        if videos_video35 is not None:
            self.videos_video35 = videos_video35
        if videos_video36 is not None:
            self.videos_video36 = videos_video36
        if videos_video37 is not None:
            self.videos_video37 = videos_video37
        if videos_video38 is not None:
            self.videos_video38 = videos_video38
        if videos_video39 is not None:
            self.videos_video39 = videos_video39
        if videos_video40 is not None:
            self.videos_video40 = videos_video40
        if videos_video41 is not None:
            self.videos_video41 = videos_video41
        if videos_video42 is not None:
            self.videos_video42 = videos_video42
        if videos_video43 is not None:
            self.videos_video43 = videos_video43
        if videos_video44 is not None:
            self.videos_video44 = videos_video44
        if videos_video45 is not None:
            self.videos_video45 = videos_video45
        if videos_video46 is not None:
            self.videos_video46 = videos_video46
        if videos_video47 is not None:
            self.videos_video47 = videos_video47
        if videos_video48 is not None:
            self.videos_video48 = videos_video48
        if videos_video49 is not None:
            self.videos_video49 = videos_video49
        if videos_video50 is not None:
            self.videos_video50 = videos_video50
        if videos_video51 is not None:
            self.videos_video51 = videos_video51
        if videos_video52 is not None:
            self.videos_video52 = videos_video52
        if videos_video53 is not None:
            self.videos_video53 = videos_video53
        if videos_video54 is not None:
            self.videos_video54 = videos_video54
        if videos_video55 is not None:
            self.videos_video55 = videos_video55
        if videos_video56 is not None:
            self.videos_video56 = videos_video56
        if videos_video57 is not None:
            self.videos_video57 = videos_video57
        if videos_video58 is not None:
            self.videos_video58 = videos_video58
        if videos_video59 is not None:
            self.videos_video59 = videos_video59
        if videos_video60 is not None:
            self.videos_video60 = videos_video60
        if videos_video61 is not None:
            self.videos_video61 = videos_video61
        if videos_video62 is not None:
            self.videos_video62 = videos_video62
        if videos_video63 is not None:
            self.videos_video63 = videos_video63
        if videos_video64 is not None:
            self.videos_video64 = videos_video64
        if videos_video65 is not None:
            self.videos_video65 = videos_video65
        if videos_video66 is not None:
            self.videos_video66 = videos_video66
        if videos_video67 is not None:
            self.videos_video67 = videos_video67
        if videos_video68 is not None:
            self.videos_video68 = videos_video68
        if videos_video69 is not None:
            self.videos_video69 = videos_video69
        if videos_video70 is not None:
            self.videos_video70 = videos_video70
        if videos_video71 is not None:
            self.videos_video71 = videos_video71
        if videos_video72 is not None:
            self.videos_video72 = videos_video72
        if videos_video73 is not None:
            self.videos_video73 = videos_video73
        if videos_video74 is not None:
            self.videos_video74 = videos_video74
        if videos_video75 is not None:
            self.videos_video75 = videos_video75
        if videos_video76 is not None:
            self.videos_video76 = videos_video76
        if videos_video77 is not None:
            self.videos_video77 = videos_video77
        if videos_video78 is not None:
            self.videos_video78 = videos_video78
        if videos_video79 is not None:
            self.videos_video79 = videos_video79
        if videos_video80 is not None:
            self.videos_video80 = videos_video80
        if videos_video81 is not None:
            self.videos_video81 = videos_video81
        if videos_video82 is not None:
            self.videos_video82 = videos_video82
        if videos_video83 is not None:
            self.videos_video83 = videos_video83
        if videos_video84 is not None:
            self.videos_video84 = videos_video84
        if videos_video85 is not None:
            self.videos_video85 = videos_video85
        if videos_video86 is not None:
            self.videos_video86 = videos_video86
        if videos_video87 is not None:
            self.videos_video87 = videos_video87
        if videos_video88 is not None:
            self.videos_video88 = videos_video88
        if videos_video89 is not None:
            self.videos_video89 = videos_video89
        if videos_video90 is not None:
            self.videos_video90 = videos_video90
        if videos_video91 is not None:
            self.videos_video91 = videos_video91
        if videos_video92 is not None:
            self.videos_video92 = videos_video92
        if videos_video93 is not None:
            self.videos_video93 = videos_video93
        if videos_video94 is not None:
            self.videos_video94 = videos_video94
        if videos_video95 is not None:
            self.videos_video95 = videos_video95
        if videos_video96 is not None:
            self.videos_video96 = videos_video96
        if videos_video97 is not None:
            self.videos_video97 = videos_video97
        if videos_video98 is not None:
            self.videos_video98 = videos_video98
        if videos_video99 is not None:
            self.videos_video99 = videos_video99
        self.codec = codec
        if complete_audio is not None:
            self.complete_audio = complete_audio
        self.__validate()

    def __validate(self) -> None:
        validate_choice("codec", self.codec, CONCATENATEVIDEO_CODECS)


from . import batch, preprocessors, transform
