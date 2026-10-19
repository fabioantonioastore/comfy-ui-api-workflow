from typing import Any, Literal, get_args

from .base import BaseNode
from .validation import validate_choice, validate_not_empty, validate_range
from .dynamic import Choice, Combo, DynamicInputs, validate_dynamic


class LoadAudio(BaseNode):
    __slots__ = ("audio",)
    CATEGORY = "audio"
    OUTPUTS = {"AUDIO": 0}
    RETURN_TYPES = ("AUDIO",)
    INPUT_SOCKET_TYPES = {"audio": "COMBO"}

    def __init__(self, node_id: str, audio: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.audio = audio
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("audio", self.audio)


class PreviewAudio(BaseNode):
    __slots__ = ("audio",)
    CATEGORY = "audio"
    OUTPUTS = {"audio": 0}
    RETURN_TYPES = ("AUDIO",)
    INPUT_SOCKET_TYPES = {"audio": "AUDIO"}

    def __init__(self, node_id: str, audio: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.audio = audio


class RecordAudio(BaseNode):
    __slots__ = ("audio",)
    CATEGORY = "audio"
    OUTPUTS = {"AUDIO": 0}
    RETURN_TYPES = ("AUDIO",)
    INPUT_SOCKET_TYPES = {"audio": "AUDIO_RECORD"}

    def __init__(self, node_id: str, audio: Any) -> None:
        super().__init__(node_id=node_id)
        self.audio = audio


class TrimAudioDuration(BaseNode):
    __slots__ = ("audio", "start_index", "duration")
    CATEGORY = "audio"
    OUTPUTS = {"AUDIO": 0}
    RETURN_TYPES = ("AUDIO",)
    INPUT_SOCKET_TYPES = {"audio": "AUDIO", "start_index": "FLOAT", "duration": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        audio: tuple[str, int],
        start_index: float | tuple[str, int] = 0.0,
        duration: float | tuple[str, int] = 60.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.audio = audio
        self.start_index = start_index
        self.duration = duration
        self.__validate()

    def __validate(self) -> None:
        validate_range(
            "start_index",
            self.start_index,
            -18_446_744_073_709_551_615,
            18_446_744_073_709_551_615,
        )
        validate_range("duration", self.duration, 0.0)


class SplitAudioChannels(BaseNode):
    __slots__ = ("audio",)
    CATEGORY = "audio"
    OUTPUTS = {"left": 0, "right": 1}
    RETURN_TYPES = ("AUDIO", "AUDIO")
    INPUT_SOCKET_TYPES = {"audio": "AUDIO"}

    def __init__(self, node_id: str, audio: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.audio = audio


class JoinAudioChannels(BaseNode):
    __slots__ = ("audio_left", "audio_right")
    CATEGORY = "audio"
    OUTPUTS = {"audio": 0}
    RETURN_TYPES = ("AUDIO",)
    INPUT_SOCKET_TYPES = {"audio_left": "AUDIO", "audio_right": "AUDIO"}

    def __init__(
        self, node_id: str, audio_left: tuple[str, int], audio_right: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.audio_left = audio_left
        self.audio_right = audio_right


AudioConcat_Directions = Literal["after", "before"]
AUDIOCONCAT_DIRECTIONS = frozenset(get_args(AudioConcat_Directions))


class AudioConcat(BaseNode):
    __slots__ = ("audio1", "audio2", "direction")
    CATEGORY = "audio"
    OUTPUTS = {"AUDIO": 0}
    RETURN_TYPES = ("AUDIO",)
    INPUT_SOCKET_TYPES = {"audio1": "AUDIO", "audio2": "AUDIO", "direction": "COMBO"}

    def __init__(
        self,
        node_id: str,
        audio1: tuple[str, int],
        audio2: tuple[str, int],
        direction: AudioConcat_Directions | tuple[str, int] = "after",
    ) -> None:
        super().__init__(node_id=node_id)
        self.audio1 = audio1
        self.audio2 = audio2
        self.direction = direction
        self.__validate()

    def __validate(self) -> None:
        validate_choice("direction", self.direction, AUDIOCONCAT_DIRECTIONS)


AudioMerge_Merge_Methods = Literal["add", "mean", "subtract", "multiply"]
AUDIOMERGE_MERGE_METHODS = frozenset(get_args(AudioMerge_Merge_Methods))


class AudioMerge(BaseNode):
    __slots__ = ("audio1", "audio2", "merge_method")
    CATEGORY = "audio"
    OUTPUTS = {"AUDIO": 0}
    RETURN_TYPES = ("AUDIO",)
    INPUT_SOCKET_TYPES = {"audio1": "AUDIO", "audio2": "AUDIO", "merge_method": "COMBO"}

    def __init__(
        self,
        node_id: str,
        audio1: tuple[str, int],
        audio2: tuple[str, int],
        merge_method: AudioMerge_Merge_Methods | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.audio1 = audio1
        self.audio2 = audio2
        self.merge_method = merge_method
        self.__validate()

    def __validate(self) -> None:
        validate_choice("merge_method", self.merge_method, AUDIOMERGE_MERGE_METHODS)


class AudioAdjustVolume(BaseNode):
    __slots__ = ("audio", "volume")
    CATEGORY = "audio"
    OUTPUTS = {"AUDIO": 0}
    RETURN_TYPES = ("AUDIO",)
    INPUT_SOCKET_TYPES = {"audio": "AUDIO", "volume": "INT"}

    def __init__(
        self, node_id: str, audio: tuple[str, int], volume: int | tuple[str, int] = 1
    ) -> None:
        super().__init__(node_id=node_id)
        self.audio = audio
        self.volume = volume
        self.__validate()

    def __validate(self) -> None:
        validate_range("volume", self.volume, -100, 100)


class EmptyAudio(BaseNode):
    __slots__ = ("duration", "sample_rate", "channels")
    CATEGORY = "audio"
    OUTPUTS = {"AUDIO": 0}
    RETURN_TYPES = ("AUDIO",)
    INPUT_SOCKET_TYPES = {"duration": "FLOAT", "sample_rate": "INT", "channels": "INT"}

    def __init__(
        self,
        node_id: str,
        duration: float | tuple[str, int] = 60.0,
        sample_rate: int | tuple[str, int] = 44_100,
        channels: int | tuple[str, int] = 2,
    ) -> None:
        super().__init__(node_id=node_id)
        self.duration = duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.__validate()

    def __validate(self) -> None:
        validate_range("duration", self.duration, 0.0, 18_446_744_073_709_551_615)
        validate_range("sample_rate", self.sample_rate, 1, 192_000)
        validate_range("channels", self.channels, 1, 2)


class AudioEqualizer3Band(BaseNode):
    __slots__ = (
        "audio",
        "low_gain_dB",
        "low_freq",
        "mid_gain_dB",
        "mid_freq",
        "mid_q",
        "high_gain_dB",
        "high_freq",
    )
    CATEGORY = "audio"
    OUTPUTS = {"AUDIO": 0}
    RETURN_TYPES = ("AUDIO",)
    INPUT_SOCKET_TYPES = {
        "audio": "AUDIO",
        "low_gain_dB": "FLOAT",
        "low_freq": "INT",
        "mid_gain_dB": "FLOAT",
        "mid_freq": "INT",
        "mid_q": "FLOAT",
        "high_gain_dB": "FLOAT",
        "high_freq": "INT",
    }

    def __init__(
        self,
        node_id: str,
        audio: tuple[str, int],
        low_gain_dB: float | tuple[str, int] = 0.0,
        low_freq: int | tuple[str, int] = 100,
        mid_gain_dB: float | tuple[str, int] = 0.0,
        mid_freq: int | tuple[str, int] = 1000,
        mid_q: float | tuple[str, int] = 0.707,
        high_gain_dB: float | tuple[str, int] = 0.0,
        high_freq: int | tuple[str, int] = 5000,
    ) -> None:
        super().__init__(node_id=node_id)
        self.audio = audio
        self.low_gain_dB = low_gain_dB
        self.low_freq = low_freq
        self.mid_gain_dB = mid_gain_dB
        self.mid_freq = mid_freq
        self.mid_q = mid_q
        self.high_gain_dB = high_gain_dB
        self.high_freq = high_freq
        self.__validate()

    def __validate(self) -> None:
        validate_range("low_gain_dB", self.low_gain_dB, -24.0, 24.0)
        validate_range("low_freq", self.low_freq, 20, 500)
        validate_range("mid_gain_dB", self.mid_gain_dB, -24.0, 24.0)
        validate_range("mid_freq", self.mid_freq, 200, 4000)
        validate_range("mid_q", self.mid_q, 0.1, 10.0)
        validate_range("high_gain_dB", self.high_gain_dB, -24.0, 24.0)
        validate_range("high_freq", self.high_freq, 1000, 15_000)


SaveAudioAdvanced_Formats = Literal["flac", "mp3", "opus"]


SaveAudioAdvanced_Format_Qualitys = Literal["V0", "128k", "320k", "64k", "96k", "192k"]


class SaveAudioAdvanced(DynamicInputs, BaseNode):
    __slots__ = (
        "audio",
        "filename_prefix",
        "format",
        "format_quality",
    )
    CATEGORY = "audio"
    OUTPUTS = {"audio": 0}
    RETURN_TYPES = ("AUDIO",)
    INPUT_NAMES = {"format_quality": "format.quality"}
    INPUT_SOCKET_TYPES = {
        "audio": "AUDIO",
        "filename_prefix": "STRING",
        "format": "COMFY_DYNAMICCOMBO_V3",
        "format_quality": "COMBO",
    }
    DYNAMIC_INPUTS = {
        "format": Combo(
            {
                "flac": {},
                "mp3": {"format_quality": Choice(("V0", "128k", "320k"), default="V0")},
                "opus": {
                    "format_quality": Choice(
                        ("64k", "96k", "128k", "192k", "320k"), default="128k"
                    )
                },
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        audio: tuple[str, int],
        filename_prefix: str | tuple[str, int] = "audio/ComfyUI",
        format: SaveAudioAdvanced_Formats = "flac",
        format_quality: (
            SaveAudioAdvanced_Format_Qualitys | tuple[str, int] | None
        ) = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.audio = audio
        self.filename_prefix = filename_prefix
        self.format = format
        if format_quality is not None:
            self.format_quality = format_quality
        self.__validate()

    def __validate(self) -> None:
        validate_dynamic(self)
