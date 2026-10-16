from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice, validate_range

YuE2GenerateABC_Modes = Literal["full", "melody"]
YUE2GENERATEABC_MODES = frozenset(get_args(YuE2GenerateABC_Modes))


class YuE2GenerateABC(BaseNode):
    __slots__ = (
        "clip",
        "style",
        "lyrics",
        "mode",
        "seed",
        "max_abc_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "penalty_window",
    )
    CATEGORY = "model/conditioning/yue2"
    OUTPUTS = {"abc": 0}
    RETURN_TYPES = ("STRING",)
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "style": "STRING",
        "lyrics": "STRING",
        "seed": "INT",
        "mode": "COMBO",
        "max_abc_tokens": "INT",
        "temperature": "FLOAT",
        "top_p": "FLOAT",
        "top_k": "INT",
        "repetition_penalty": "FLOAT",
        "penalty_window": "INT",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        style: str | tuple[str, int],
        lyrics: str | tuple[str, int],
        mode: YuE2GenerateABC_Modes | tuple[str, int],
        seed: int | tuple[str, int] = 0,
        max_abc_tokens: int | tuple[str, int] = 8192,
        temperature: float | tuple[str, int] = 0.7,
        top_p: float | tuple[str, int] = 0.9,
        top_k: int | tuple[str, int] = 30,
        repetition_penalty: float | tuple[str, int] = 1.005,
        penalty_window: int | tuple[str, int] = 100,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.style = style
        self.lyrics = lyrics
        self.mode = mode
        self.seed = seed
        self.max_abc_tokens = max_abc_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.penalty_window = penalty_window
        self.__validate()

    def __validate(self) -> None:
        validate_choice("mode", self.mode, YUE2GENERATEABC_MODES)
        validate_range("seed", self.seed, 0, 18_446_744_073_709_551_615)
        validate_range("max_abc_tokens", self.max_abc_tokens, 1, 20_000)
        validate_range("temperature", self.temperature, 0.0, 5.0)
        validate_range("top_p", self.top_p, 0.01, 1.0)
        validate_range("top_k", self.top_k, 1, 32_768)
        validate_range("repetition_penalty", self.repetition_penalty, 0.01, 10.0)
        validate_range("penalty_window", self.penalty_window, 1, 20_000)


YuE2GenerateMusic_Modes = Literal["full", "melody"]
YUE2GENERATEMUSIC_MODES = frozenset(get_args(YuE2GenerateMusic_Modes))


class YuE2GenerateMusic(BaseNode):
    __slots__ = (
        "clip",
        "style",
        "lyrics",
        "mode",
        "abc",
        "seed",
        "max_duration",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "cfg_scale",
    )
    CATEGORY = "model/conditioning/yue2"
    OUTPUTS = {"CONDITIONING": 0, "seconds": 1}
    RETURN_TYPES = ("CONDITIONING", "FLOAT")
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "style": "STRING",
        "lyrics": "STRING",
        "abc": "STRING",
        "seed": "INT",
        "mode": "COMBO",
        "max_duration": "FLOAT",
        "temperature": "FLOAT",
        "top_p": "FLOAT",
        "top_k": "INT",
        "repetition_penalty": "FLOAT",
        "cfg_scale": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        style: str | tuple[str, int],
        lyrics: str | tuple[str, int],
        mode: YuE2GenerateMusic_Modes | tuple[str, int],
        abc: str | tuple[str, int] = "",
        seed: int | tuple[str, int] = 0,
        max_duration: float | tuple[str, int] = 360.0,
        temperature: float | tuple[str, int] = 1.0,
        top_p: float | tuple[str, int] = 0.95,
        top_k: int | tuple[str, int] = 100,
        repetition_penalty: float | tuple[str, int] = 1.2,
        cfg_scale: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.style = style
        self.lyrics = lyrics
        self.mode = mode
        self.abc = abc
        self.seed = seed
        self.max_duration = max_duration
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.cfg_scale = cfg_scale
        self.__validate()

    def __validate(self) -> None:
        validate_choice("mode", self.mode, YUE2GENERATEMUSIC_MODES)
        validate_range("seed", self.seed, 0, 18_446_744_073_709_551_615)
        validate_range("max_duration", self.max_duration, 0.04, 900.0)
        validate_range("temperature", self.temperature, 0.0, 5.0)
        validate_range("top_p", self.top_p, 0.01, 1.0)
        validate_range("top_k", self.top_k, 1, 32_768)
        validate_range("repetition_penalty", self.repetition_penalty, 0.01, 10.0)
        validate_range("cfg_scale", self.cfg_scale, 0.0, 100.0)


SheetSage2AudioToABC_Modes = Literal["melody", "full"]
SHEETSAGE2AUDIOTOABC_MODES = frozenset(get_args(SheetSage2AudioToABC_Modes))


class SheetSage2AudioToABC(BaseNode):
    __slots__ = ("audio_encoder", "audio", "mode")
    CATEGORY = "model/conditioning/yue2"
    OUTPUTS = {"abc": 0}
    RETURN_TYPES = ("STRING",)
    INPUT_SOCKET_TYPES = {
        "audio_encoder": "AUDIO_ENCODER",
        "audio": "AUDIO",
        "mode": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        audio_encoder: tuple[str, int],
        audio: tuple[str, int],
        mode: SheetSage2AudioToABC_Modes | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.audio_encoder = audio_encoder
        self.audio = audio
        self.mode = mode
        self.__validate()

    def __validate(self) -> None:
        validate_choice("mode", self.mode, SHEETSAGE2AUDIOTOABC_MODES)
